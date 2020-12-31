import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_scatter import scatter_mean, scatter_add


class RNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers,
                            batch_first=True)

    def forward(self, x, h0, c0):
        # Set initial hidden and cell states
        # Forward propagate LSTM
        out, (h_n, c_n) = self.lstm(x, (h0, c0))  
        # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        return h_n, c_n


class RNNDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNNDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers,
                            batch_first=True)
        self.lin = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x, h_0, c_0):
        # Forward propagate LSTM
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = self.lin(out)
        out = self.relu(out)
        # Decode the hidden state of the last time step
        return out, h_n, c_n


class GlobalModel(torch.nn.Module):
    def __init__(self, aqi_em, rnn_h, rnn_l, gnn_h):
        super(GlobalModel, self).__init__()
        self.aqi_em = aqi_em
        self.rnn_h = rnn_h
        self.gnn_h = gnn_h
        self.aqi_embed = Seq(Lin(1, aqi_em), ReLU())
        self.aqi_rnn = nn.LSTM(aqi_em, rnn_h, rnn_l, batch_first=True)
        self.city_gnn = CityGNN(rnn_h, 2, gnn_h)

    def batchInput(self, x, edge_w, edge_conn):
        sta_num = x.shape[1]
        x = x.reshape(-1, x.shape[-1])
        edge_w = edge_w.reshape(-1, edge_w.shape[-1])
        for i in range(edge_conn.size(0)):
            edge_conn[i, :] = torch.add(edge_conn[i, :], i * sta_num)
        edge_conn = edge_conn.transpose(0, 1)
        edge_conn = edge_conn.reshape(2, -1)
        return x, edge_w, edge_conn

    def forward(self, city_aqi, city_conn, city_w, city_num):
        city_aqi = city_aqi.unsqueeze(dim=-1)
        city_aqi = self.aqi_embed(city_aqi)
        city_aqi, _ = self.aqi_rnn(city_aqi.reshape(-1, 24, self.aqi_em))
        city_aqi = city_aqi.reshape(-1, 10, 24, self.rnn_h)
        city_aqi = city_aqi.transpose(1, 2)
        city_aqi = city_aqi.reshape(-1, city_num, city_aqi.shape[-1])

        city_conn = city_conn.transpose(1, 2).repeat(24, 1, 1)
        city_w = city_w.reshape(-1, city_w.shape[-2], city_w.shape[-1])
        # print(city_aqi.shape,city_conn.shape, city_w.shape)
        city_x, city_weight, city_conn = self.batchInput(
            city_aqi, city_w, city_conn)
        out = self.city_gnn(city_x, city_conn, city_weight)
        out = out.reshape(-1, 24, city_num, out.shape[-1])

        return out


class CityGNN(torch.nn.Module):
    def __init__(self, node_h, edge_h, gnn_h):
        super(CityGNN, self).__init__()
        self.node_mlp_1 = Seq(Lin(2 * node_h + edge_h, gnn_h),
                              ReLU(inplace=True))
        self.node_mlp_2 = Seq(Lin(node_h + gnn_h, gnn_h), ReLU(inplace=True))

    def forward(self, x, edge_index, edge_attr):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        row, col = edge_index
        out = torch.cat([x[row], x[col], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)
        return self.node_mlp_2(out)


class CityModel(nn.Module):
    """Station graph"""
    def __init__(self, aqi_em, poi_em, wea_em, rnn_h, rnn_l, gnn_h):
        super(CityModel, self).__init__()
        self.rnn_h = rnn_h
        self.gnn_h = gnn_h
        self.rnn_l = rnn_l
        self.aqi_embed = Seq(Lin(1, aqi_em), ReLU())
        self.poi_embed = Seq(Lin(5, poi_em), ReLU())
        self.city_embed = Seq(Lin(gnn_h, wea_em), ReLU())
        self.wea_embed = Seq(Lin(5, wea_em), ReLU())
        self.sta_gnn = StaGNN(aqi_em + poi_em, 2, gnn_h, 2 * wea_em)
        self.encoder = RNNEncoder(input_size=gnn_h,
                                  hidden_size=rnn_h,
                                  num_layers=rnn_l)
        self.decoder_embed = Seq(Lin(1, aqi_em), ReLU())
        self.decoder = RNNDecoder(input_size=4 + aqi_em,
                                  hidden_size=rnn_h,
                                  num_layers=rnn_l)

    def batchInput(self, x, edge_w, edge_conn):
        sta_num = x.shape[1]
        x = x.reshape(-1, x.shape[-1])
        edge_w = edge_w.reshape(-1, edge_w.shape[-1])
        for i in range(edge_conn.size(0)):
            edge_conn[i, :] = torch.add(edge_conn[i, :], i * sta_num)
        edge_conn = edge_conn.transpose(0, 1)
        edge_conn = edge_conn.reshape(2, -1)
        return x, edge_w, edge_conn

    def forward(self, city_data, city_u, device):
        sta_aqi, sta_conn, sta_poi, sta_w, sta_wea, sta_for, _ = city_data
        sta_num = sta_aqi.shape[1]
        sta_x = sta_aqi.unsqueeze(dim=-1)
        sta_x = self.aqi_embed(sta_x)
        sta_poi = self.poi_embed(sta_poi)
        sta_poi = sta_poi.unsqueeze(dim=-2).repeat_interleave(24, dim=-2)
        sta_x = torch.cat([sta_x, sta_poi], dim=-1)
        sta_x = sta_x.transpose(1, 2)
        sta_x = sta_x.reshape(-1, sta_x.shape[-2], sta_x.shape[-1])

        sta_conn = sta_conn.transpose(1, 2).repeat(24, 1, 1)
        sta_w = sta_w.reshape(-1, sta_w.shape[-2], sta_w.shape[-1])
        # print(sta_x.shape,sta_conn.shape,sta_w.shape)
        sta_x, sta_weight, sta_conn = self.batchInput(sta_x, sta_w, sta_conn)
        city_u = self.city_embed(city_u)
        sta_wea = self.wea_embed(sta_wea)
        sta_u = torch.cat([city_u, sta_wea], dim=-1)
        sta_x = self.sta_gnn(sta_x, sta_conn, sta_weight, sta_u, sta_num)
        sta_x = sta_x.reshape(-1, 24, sta_num, sta_x.shape[-1]).transpose(1, 2)
        sta_x = sta_x.reshape(-1, 24, sta_x.shape[-1])

        h0 = torch.randn(self.rnn_l, sta_x.size(0), self.rnn_h).to(device)
        c0 = torch.randn(self.rnn_l, sta_x.size(0), self.rnn_h).to(device)
        h_x, c_x = self.encoder(sta_x, h0, c0)

        outputs = torch.zeros((sta_x.size(0), sta_for.size(1), 1)).to(device)
        aqi = sta_aqi[:, :, -1].reshape(-1, 1)
        sta_for = sta_for.repeat(sta_num, 1, 1)
        for i in range(sta_for.size(1)):
            aqi_em = self.decoder_embed(aqi)
            inputs = torch.cat((aqi_em, sta_for[:, i]), dim=-1)
            inputs = inputs.unsqueeze(dim=1)
            output, h_x, c_x = self.decoder(inputs, h_x, c_x)
            output = output.reshape(-1, 1)
            outputs[:, i] = output
            aqi = output
        outputs = outputs.reshape(-1, sta_num, sta_for.size(1))

        return outputs


class StaGNN(torch.nn.Module):
    def __init__(self, node_h, edge_h, gnn_h, u_h):
        super(StaGNN, self).__init__()
        self.node_mlp_1 = Seq(Lin(2 * node_h + edge_h, gnn_h),
                              ReLU(inplace=True))
        self.node_mlp_2 = Seq(Lin(node_h + gnn_h + u_h, gnn_h),
                              ReLU(inplace=True))

    def forward(self, x, edge_index, edge_attr, u, sta_num):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        u = u.reshape(-1, u.shape[-1])
        u = u.repeat(sta_num, 1)
        row, col = edge_index
        out = torch.cat([x[row], x[col], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out, u], dim=1)
        return self.node_mlp_2(out)