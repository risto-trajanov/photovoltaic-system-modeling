import requests
import pandas as pd
import json
from utils import persist_data
from datetime import datetime
import numpy

key = '6a1bfa1b5b41497c94ea0d86aee73671'
lat = 51.1877
long = 10.0398


# 'lat=51.1877&lon=10.0398&start_date=2020-01-01&end_date=2020-02-01&key=6a1bfa1b5b41497c94ea0d86aee73671'

class Crawl:
    def __init__(self, lat, long):
        self.lat = lat
        self.long = long
        dataframe = self.to_DataFrame(5, 6)
        forecast = self.getForecast()
        self.persist_data('tsdb_weatherbit_cleaned', dataframe)
        self.persist_data('tsdb_weather_forecast', forecast)

    def to_DataFrame(self, from_month, to_month):
        dataframe = []
        api = 'https://api.weatherbit.io/v2.0/history/hourly?'
        for month in range(from_month, to_month):
            for day in range(1, 30):
                resp = requests.get(
                    api + f'lat={lat}&lon={long}&start_date=2020-{month:02d}-{day:02d}&end_date=2020-{month:02d}-{day + 1:02d}&key={key}')
                json_resp = json.loads(resp.text)
                data = json_resp['data']
                df_rows = []
                for hour_data in data:
                    date_time = hour_data['timestamp_utc']
                    date_time = date_time.replace('T', ' ')
                    datetime_object = datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S')
                    gh = hour_data['solar_rad']
                    csky_ghi = hour_data['ghi']
                    tdry = hour_data['temp']
                    row = [datetime_object, gh, csky_ghi, tdry]
                    df_rows.append(row)
                dataframe.append(df_rows)

        data_to_db = pd.DataFrame(numpy.concatenate(dataframe), columns=['datetime', 'gh', 'csky_ghi', 'tdry'])
        data_to_db.set_index('datetime', inplace=True)
        return data_to_db

    def persist_data(self, table, dataframe):
        persist_data(table, dataframe)

    def getForecast(self):
        api = 'https://api.weatherbit.io/v2.0/forecast/hourly?'
        resp = requests.get(api + f'lat={lat}&lon={long}&key={key}')
        json_resp = json.loads(resp.text)
        data = json_resp['data']
        dataframe = []
        df_rows = []
        for hour_data in data:
            date_time = hour_data['timestamp_utc']
            date_time = date_time.replace('T', ' ')
            datetime_object = datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S')
            gh = hour_data['solar_rad']
            csky_ghi = hour_data['ghi']
            tdry = hour_data['temp']
            row = [datetime_object, gh, csky_ghi, tdry]
            df_rows.append(row)
        dataframe.append(df_rows)

        data_to_db = pd.DataFrame(numpy.concatenate(dataframe), columns=['datetime', 'gh', 'csky_ghi', 'tdry'])
        data_to_db.set_index('datetime', inplace=True)
        return data_to_db


def main():
    crawler = Crawl(lat, long)


