import requests
import pandas as pd
import json
from utils import persist_data
from utils import read_data_for_day
from utils import download_s3
from datetime import datetime
from datetime import date
import numpy
import os
from arima import Transform
import pickle

exogenous_features = ["gh_mean_lag3", "gh_std_lag3", "tdry_mean_lag3", "tdry_std_lag3", "csky_ghi_mean_lag3",
                      "csky_ghi_std_lag3",
                      "gh_mean_lag7", "gh_std_lag7", "tdry_mean_lag7", "tdry_std_lag7",
                      "csky_ghi_mean_lag7", "csky_ghi_std_lag7",
                      "gh_mean_lag30", "gh_std_lag30", "tdry_mean_lag30", "tdry_std_lag30",
                      "csky_ghi_mean_lag30", "csky_ghi_std_lag30",
                      "month", "week", "day", "day_of_week"]
key = '6a1bfa1b5b41497c94ea0d86aee73671'
lat = 51.1877
long = 10.0398
api = 'https://api.weatherbit.io/v2.0/forecast/hourly?'


# lat=51.1877&lon=10.0398&key=6a1bfa1b5b41497c94ea0d86aee73671


class Predict:
    def __init__(self, target, table):
        self.table = table
        self.model_from_s3(target)
        predictions = self.predictions_today(target)
        self.persist_predictions(target, predictions)

    def models(self, target):
        arima = 'arima_model_' + target
        prophet = 'prophet_model_' + target
        latest = date(1999, 1, 1)
        for file in os.listdir("./Models"):
            if file.endswith(".pickle"):
                name = os.path.splitext(file)[0]
                split = name.split("_")
                if split[0].startswith("temp"):
                    continue
                date_str = split[-3:]
                year = int(date_str[2])
                month = int(date_str[1])
                day = int(date_str[0])
                file_date = date(year, month, day)
                if file_date > latest:
                    latest = file_date

        arima = arima + f'_{latest.day:02d}_{latest.month:02d}_{latest.year}.pickle'
        prophet = prophet + f'_{latest.day:02d}_{latest.month:02d}_{latest.year}.pickle'

        with open("./Models/" + arima, 'rb') as arima:
            self.arima_model = pickle.load(arima)

        # with open("./Models/" + prophet, 'rb') as prophet:
        #     self.prophet_model = pickle.load(prophet)

    def model_from_s3(self, target):
        arima = 'temp_arima_model_' + target
        now = date.today()
        arima = arima + f'_{now.day:02d}_{now.month:02d}_{now.year}.pickle'
        s3_path = os.path.join('Models', arima)
        print(s3_path)
        model = download_s3(s3_path)
        with open(model, 'rb') as pickle_model:
            self.arima_model = pickle.load(pickle_model)

    def predictions_today(self, target):
        now = date.today()
        data = read_data_for_day(self.table, str(now))
        data_for_prediction = self.prepare_data(data)

        # prophet_predictions = self.prophet_model.predict(
        #     data_for_prediction[['datetime'] + exogenous_features].rename(columns={'datetime': "ds"}))

        arima_predictions = self.arima_model.predict(n_periods=len(data_for_prediction),
                                                     exogenous=data[exogenous_features])

        arima_predictions = arima_predictions.T
        datetime = data_for_prediction['datetime']
        datetime = pd.DataFrame(datetime)
        datetime[target] = arima_predictions
        datetime.drop('datetime', 1, inplace=True)

        predictions = datetime
        self.predictions = predictions

        return predictions

    def persist_predictions(self, target, predictions):
        table = 'tsdb_' + target + '_predictions'
        persist_data(table, predictions)
        print("persisted")

    def prepare_data(self, data):
        arima_transform = Transform(data)
        return arima_transform.get_data()


def main():
    table = 'tsdb_weather_forecast'
    target = 'consumption'
    prediction_consumption = Predict(target, table)
    table = 'tsdb_weather_forecast'
    target = 'energy'
    prediction_consumption = Predict(target, table)

