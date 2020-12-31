import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data

import os
import json
import numpy as np
import pandas as pd

TIME_WINDOW = 24
PRED_TIME = 12



DATA_PATH = './data'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class trainDataset(Data.Dataset):
    def __init__(self, transform=None, train=True):
        with open(os.path.join(DATA_PATH,'data','city_train.txt'), 'r') as f:
            self.cities = json.load(f)

        with open(os.path.join(DATA_PATH,'data','jiaxing_train.txt'), 'r') as f:
            self.jiaxing = json.load(f)

        with open(os.path.join(DATA_PATH,'data','shanghai_train.txt'), 'r') as f:
            self.shanghai = json.load(f)

        with open(os.path.join(DATA_PATH,'data','suzhou_train.txt'), 'r') as f:
            self.suzhou = json.load(f)

        with open(os.path.join(DATA_PATH,'stations.txt'), 'r') as f:
            self.stations = json.load(f)

    def GetCityData(self,city_name,city_source,index):
        station_list = self.stations[city_name]
        city_aqi = []
        city_y = []
        for x in station_list:
            city_aqi.append(city_source[x][index][:TIME_WINDOW])
            city_y.append(city_source[x][index][TIME_WINDOW:])

        city_aqi = torch.FloatTensor(city_aqi)
        city_y = torch.FloatTensor(city_y)
        city_sim = torch.FloatTensor(city_source['sim'][index])
        city_conn = torch.tensor(city_source['conn'])
        city_weather = torch.FloatTensor(city_source['weather'][index])
        city_for = torch.FloatTensor(city_source['weather_for'][index])
        city_poi = torch.FloatTensor(city_source['poi'])

        city_data = [city_aqi, city_conn, city_poi, city_sim,
                city_weather, city_for, city_y]
		
        return city_data


    def __getitem__(self, index):
        jiaxing_data = self.GetCityData('jiaxing',self.jiaxing,index)
        shanghai_data = self.GetCityData('shanghai',self.shanghai,index)
        suzhou_data = self.GetCityData('suzhou',self.suzhou,index)

        cities_aqi = torch.FloatTensor(self.cities['aqi'][index])
        cities_conn = torch.tensor(self.cities['conn'])
        cities_weather = torch.FloatTensor(self.cities['weather'][index])
        cities_sim = torch.FloatTensor(self.cities['sim'][index])

        cities_data = [cities_aqi, cities_conn,cities_sim,cities_weather]
        
        return cities_data,jiaxing_data,shanghai_data,suzhou_data

    def __len__(self):
        return len(self.shanghai['weather'])

	

class valDataset(Data.Dataset):
    def __init__(self, transform=None, train=True):
        with open(os.path.join(DATA_PATH,'data','city_val.txt'), 'r') as f:
            self.cities = json.load(f)

        with open(os.path.join(DATA_PATH,'data','jiaxing_val.txt'), 'r') as f:
            self.jiaxing = json.load(f)

        with open(os.path.join(DATA_PATH,'data','shanghai_val.txt'), 'r') as f:
            self.shanghai = json.load(f)

        with open(os.path.join(DATA_PATH,'data','suzhou_val.txt'), 'r') as f:
            self.suzhou = json.load(f)

        with open(os.path.join(DATA_PATH,'stations.txt'), 'r') as f:
            self.stations = json.load(f)

    def GetCityData(self,city_name,city_source,index):
        station_list = self.stations[city_name]
        city_aqi = []
        city_y = []
        for x in station_list:
            city_aqi.append(city_source[x][index][:TIME_WINDOW])
            city_y.append(city_source[x][index][TIME_WINDOW:])

        city_aqi = torch.FloatTensor(city_aqi)
        city_y = torch.FloatTensor(city_y)
        city_sim = torch.FloatTensor(city_source['sim'][index])
        city_conn = torch.tensor(city_source['conn'])
        city_weather = torch.FloatTensor(city_source['weather'][index])
        city_for = torch.FloatTensor(city_source['weather_for'][index])
        city_poi = torch.FloatTensor(city_source['poi'])

        city_data = [city_aqi, city_conn, city_poi, city_sim,
                city_weather, city_for, city_y]
		
        return city_data


    def __getitem__(self, index):
        jiaxing_data = self.GetCityData('jiaxing',self.jiaxing,index)
        shanghai_data = self.GetCityData('shanghai',self.shanghai,index)
        suzhou_data = self.GetCityData('suzhou',self.suzhou,index)

        cities_aqi = torch.FloatTensor(self.cities['aqi'][index])
        cities_conn = torch.tensor(self.cities['conn'])
        cities_weather = torch.FloatTensor(self.cities['weather'][index])
        cities_sim = torch.FloatTensor(self.cities['sim'][index])

        cities_data = [cities_aqi, cities_conn,cities_sim,cities_weather]
        
        return cities_data,jiaxing_data,shanghai_data,suzhou_data

    def __len__(self):
        return len(self.shanghai['weather'])

class testDataset(Data.Dataset):
    def __init__(self, transform=None, train=True):
        with open(os.path.join(DATA_PATH,'data','city_test.txt'), 'r') as f:
            self.cities = json.load(f)

        with open(os.path.join(DATA_PATH,'data','jiaxing_test.txt'), 'r') as f:
            self.jiaxing = json.load(f)

        with open(os.path.join(DATA_PATH,'data','shanghai_test.txt'), 'r') as f:
            self.shanghai = json.load(f)

        with open(os.path.join(DATA_PATH,'data','suzhou_test.txt'), 'r') as f:
            self.suzhou = json.load(f)

        with open(os.path.join(DATA_PATH,'stations.txt'), 'r') as f:
            self.stations = json.load(f)

    def GetCityData(self,city_name,city_source,index):
        station_list = self.stations[city_name]
        city_aqi = []
        city_y = []
        for x in station_list:
            city_aqi.append(city_source[x][index][:TIME_WINDOW])
            city_y.append(city_source[x][index][TIME_WINDOW:])

        city_aqi = torch.FloatTensor(city_aqi)
        city_y = torch.FloatTensor(city_y)
        city_sim = torch.FloatTensor(city_source['sim'][index])
        city_conn = torch.tensor(city_source['conn'])
        city_weather = torch.FloatTensor(city_source['weather'][index])
        city_for = torch.FloatTensor(city_source['weather_for'][index])
        city_poi = torch.FloatTensor(city_source['poi'])

        city_data = [city_aqi, city_conn, city_poi, city_sim,
                city_weather, city_for, city_y]
		
        return city_data


    def __getitem__(self, index):
        jiaxing_data = self.GetCityData('jiaxing',self.jiaxing,index)
        shanghai_data = self.GetCityData('shanghai',self.shanghai,index)
        suzhou_data = self.GetCityData('suzhou',self.suzhou,index)

        cities_aqi = torch.FloatTensor(self.cities['aqi'][index])
        cities_conn = torch.tensor(self.cities['conn'])
        cities_weather = torch.FloatTensor(self.cities['weather'][index])
        cities_sim = torch.FloatTensor(self.cities['sim'][index])

        cities_data = [cities_aqi, cities_conn,cities_sim,cities_weather]
        
        return cities_data,jiaxing_data,shanghai_data,suzhou_data

    def __len__(self):
        return len(self.shanghai['weather'])

