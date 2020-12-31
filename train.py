import argparse
import json
import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data

from dataset import testDataset, trainDataset, valDataset
from model import CityModel, GlobalModel
from torch_geometric.nn import MetaLayer

parser = argparse.ArgumentParser(description='Multi-city AQI forecasting')
parser.add_argument('--device', type=str, default='cuda', help='')
parser.add_argument('--run_times', type=int, default=5, help='')
parser.add_argument('--epoch', type=int, default=300, help='')
parser.add_argument('--batch_size', type=int, default=128, help='')
parser.add_argument('--city_num', type=int, default=10, help='')
parser.add_argument('--gnn_h', type=int, default=32, help='')
parser.add_argument('--rnn_h', type=int, default=64, help='')
parser.add_argument('--rnn_l', type=int, default=1, help='')
parser.add_argument('--aqi_em', type=int, default=16, help='')
parser.add_argument('--poi_em', type=int, default=8, help='poi embedding')
parser.add_argument('--wea_em', type=int, default=12, help='wea embedding')
parser.add_argument('--lr', type=float, default=0.001, help='lr')
parser.add_argument('--wd', type=float, default=0.001, help='weight decay')
parser.add_argument('--pred_step', type=int, default=12, help='step')
args = parser.parse_args()

device = args.device

train_dataset = trainDataset()
train_loader = Data.DataLoader(train_dataset,
                               batch_size=args.batch_size,
                               num_workers=4,
                               shuffle=True)

val_dataset = valDataset()
val_loader = Data.DataLoader(val_dataset,
                             batch_size=args.batch_size,
                             num_workers=4,
                             shuffle=True)

test_dataset = testDataset()
test_loader = Data.DataLoader(test_dataset,
                              batch_size=args.batch_size,
                              num_workers=4,
                              shuffle=False)

for runtimes in range(args.run_times):

    global_model = GlobalModel(args.aqi_em, args.rnn_h, args.rnn_l,
                               args.gnn_h).to(device)
    jiaxing_model = CityModel(args.aqi_em, args.poi_em, args.wea_em,
                              args.rnn_h, args.rnn_l, args.gnn_h).to(device)
    shanghai_model = CityModel(args.aqi_em, args.poi_em, args.wea_em,
                               args.rnn_h, args.rnn_l, args.gnn_h).to(device)
    suzhou_model = CityModel(args.aqi_em, args.poi_em, args.wea_em, args.rnn_h,
                             args.rnn_l, args.gnn_h).to(device)

    city_model_num = sum(p.numel() for p in global_model.parameters()
                         if p.requires_grad)
    print('city_model:', 'Trainable,', city_model_num)

    shanghai_model_num = sum(p.numel() for p in shanghai_model.parameters()
                             if p.requires_grad)
    print('shanghai_model_num:', 'Trainable,', shanghai_model_num)

    criterion = nn.MSELoss()
    params = list(global_model.parameters()) + list(jiaxing_model.parameters()) + \
        list(shanghai_model.parameters()) + list(suzhou_model.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)

    val_loss_min = np.inf
    for epoch in range(args.epoch):
        for i, (cities_data, jiaxing_data, shanghai_data,
                suzhou_data) in enumerate(train_loader):
            cities_aqi, cities_conn, cities_sim, _ = [x.to(device) for x in cities_data]
            # print(cities_aqi.shape, cities_conn.shape,cities_sim.shape,cities_weather.shape)
            city_u = global_model(cities_aqi, cities_conn, cities_sim,
                                  args.city_num)

            jiaxing_data = [item.to(device, non_blocking=True) for item in jiaxing_data]
            jiaxing_outputs = jiaxing_model(jiaxing_data, city_u[:, :, 4], device)
            jiaxing_loss = criterion(jiaxing_outputs, jiaxing_data[-1])

            shanghai_data = [item.to(device, non_blocking=True) for item in shanghai_data]
            shanghai_outputs = shanghai_model(shanghai_data, city_u[:, :, 6], device)
            shanghai_loss = criterion(shanghai_outputs, shanghai_data[-1])

            suzhou_data = [item.to(device, non_blocking=True) for item in suzhou_data]
            suzhou_outputs = suzhou_model(suzhou_data, city_u[:, :, 8], device)
            suzhou_loss = criterion(suzhou_outputs, suzhou_data[-1])

            jiaxing_model.zero_grad()
            shanghai_model.zero_grad()
            suzhou_model.zero_grad()
            global_model.zero_grad()

            loss = jiaxing_loss + shanghai_loss + suzhou_loss

            loss.backward()
            optimizer.step()

            if i % 20 == 0 and epoch % 50 == 0:

                print('{},Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    'jiaxing', epoch, args.epoch, i,
                    int(5922 / args.batch_size), jiaxing_loss.item()))

                print('{},Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    'shanghai', epoch, args.epoch, i,
                    int(5922 / args.batch_size), shanghai_loss.item()))

                print('{},Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    'suzhou', epoch, args.epoch, i,
                    int(5922 / args.batch_size), suzhou_loss.item()))

        val_loss = 0
        with torch.no_grad():
            for j, (cities_data_val, jiaxing_data_val, shanghai_data_val,
                    suzhou_data_val) in enumerate(val_loader):
                cities_aqi_val, cities_conn_val, cities_sim_val, _ = [x.to(device) for x in cities_data_val]
                # print(cities_aqi.shape, cities_conn.shape,cities_sim.shape,cities_weather.shape)
                city_u_val = global_model(cities_aqi_val, cities_conn_val,
                                          cities_sim_val, args.city_num)

                jiaxing_data_val = [item.to(device, non_blocking=True) for item in jiaxing_data_val]
                jiaxing_outputs_val = jiaxing_model(jiaxing_data_val, city_u_val[:, :, 4], device)
                jiaxing_loss_val = criterion(jiaxing_outputs_val, jiaxing_data_val[-1])

                shanghai_data_val = [item.to(device, non_blocking=True) for item in shanghai_data_val]
                shanghai_outputs_val = shanghai_model(shanghai_data_val, city_u_val[:, :, 6], device)                                                               
                shanghai_loss_val = criterion(shanghai_outputs_val, shanghai_data_val[-1])

                suzhou_data_val = [item.to(device, non_blocking=True) for item in suzhou_data_val]
                suzhou_outputs_val = suzhou_model(suzhou_data_val, city_u_val[:, :, 8], device)
                suzhou_loss_val = criterion(suzhou_outputs_val, suzhou_data_val[-1])

                val_loss = val_loss + jiaxing_loss_val.item(
                ) + shanghai_loss_val.item() + suzhou_loss_val.item()

            if val_loss < val_loss_min and epoch > (args.epoch * 0.7):
                torch.save(global_model.state_dict(),
                           './checkpoints/global.ckpt')
                torch.save(jiaxing_model.state_dict(),
                           './checkpoints/jiaxing.ckpt')
                torch.save(shanghai_model.state_dict(),
                           './checkpoints/shanghai.ckpt')
                torch.save(suzhou_model.state_dict(),
                           './checkpoints/suzhou.ckpt')
                val_loss_min = val_loss

    print('Finished Training')

    mae_loss = torch.zeros(3, 4)
    rmse_loss = torch.zeros(3, 4)

    def cal_loss(outputs, y, index):

        temp_loss = torch.abs(outputs - y)
        mae_loss_1 = temp_loss[:, :, 0]
        mae_loss_3 = temp_loss[:, :, 2]
        mae_loss_6 = temp_loss[:, :, 5]
        mae_loss_12 = temp_loss[:, :, -1]

        mae_loss[index, 0] += mae_loss_1.sum().item()
        mae_loss[index, 1] += mae_loss_3.sum().item()
        mae_loss[index, 2] += mae_loss_6.sum().item()
        mae_loss[index, 3] += mae_loss_12.sum().item()

        temp_loss = torch.pow(temp_loss, 2)
        rmse_loss_1 = temp_loss[:, :, 0]
        rmse_loss_3 = temp_loss[:, :, 2]
        rmse_loss_6 = temp_loss[:, :, 5]
        rmse_loss_12 = temp_loss[:, :, -1]

        rmse_loss[index, 0] += rmse_loss_1.sum().item()
        rmse_loss[index, 1] += rmse_loss_3.sum().item()
        rmse_loss[index, 2] += rmse_loss_6.sum().item()
        rmse_loss[index, 3] += rmse_loss_12.sum().item()

    with torch.no_grad():
        global_model.load_state_dict(torch.load('./checkpoints/global.ckpt'))
        jiaxing_model.load_state_dict(torch.load('./checkpoints/jiaxing.ckpt'))
        shanghai_model.load_state_dict(torch.load('./checkpoints/shanghai.ckpt'))
        suzhou_model.load_state_dict(torch.load('./checkpoints/suzhou.ckpt'))

        for i, (cities_data, jiaxing_data, shanghai_data,
                suzhou_data) in enumerate(test_loader):
            cities_aqi, cities_conn, cities_sim, _ = [x.to(device) for x in cities_data]
            city_u = global_model(cities_aqi, cities_conn, cities_sim,
                                  args.city_num)

            jiaxing_data = [item.to(device, non_blocking=True) for item in jiaxing_data]
            jiaxing_outputs = jiaxing_model(jiaxing_data, city_u[:, :, 4], device)

            shanghai_data = [item.to(device, non_blocking=True) for item in shanghai_data]
            shanghai_outputs = shanghai_model(shanghai_data, city_u[:, :, 6], device)

            suzhou_data = [item.to(device, non_blocking=True) for item in suzhou_data]
            suzhou_outputs = suzhou_model(suzhou_data, city_u[:, :, 8], device)

            cal_loss(jiaxing_outputs, jiaxing_data[-1], 0)
            cal_loss(shanghai_outputs, shanghai_data[-1], 1)
            cal_loss(suzhou_outputs, suzhou_data[-1], 2)

        mae_loss = mae_loss.numpy()
        jiaxing_mae_loss = mae_loss[0] / (len(test_dataset) * 2)
        shanghai_mae_loss = mae_loss[1] / (len(test_dataset) * 10)
        suzhou_mae_loss = mae_loss[2] / (len(test_dataset) * 8)

        jiaxing_rmse_loss = torch.sqrt(rmse_loss[0] / (len(test_dataset) * 2))
        shanghai_rmse_loss = torch.sqrt(rmse_loss[1] / (len(test_dataset) * 10))
        suzhou_rmse_loss = torch.sqrt(rmse_loss[2] / (len(test_dataset) * 8))

        print('jiaxing_mae:', jiaxing_mae_loss)
        print('shanghai_mae:', shanghai_mae_loss)
        print('suzhou_mae:', suzhou_mae_loss)

        print('jiaxing_rmse:', jiaxing_rmse_loss)
        print('shanghai_rmse:', shanghai_rmse_loss)
        print('suzhou_rmse:', suzhou_rmse_loss)

        all_mae_loss = (jiaxing_mae_loss + shanghai_mae_loss +
                        suzhou_mae_loss) / 3
        all_rmse_loss = (jiaxing_rmse_loss + shanghai_rmse_loss +
                         suzhou_rmse_loss) / 3

        print('all_mae:', all_mae_loss)
        print('all_rmse:', all_rmse_loss)
