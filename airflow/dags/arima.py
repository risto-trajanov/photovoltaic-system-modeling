import datetime

import numpy as np
import pandas as pd
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import os
from utils import read_data
#from utils import upload_s3

exogenous_features = ["gh_mean_lag3", "gh_std_lag3", "tdry_mean_lag3", "tdry_std_lag3", "csky_ghi_mean_lag3",
                      "csky_ghi_std_lag3",
                      "gh_mean_lag7", "gh_std_lag7", "tdry_mean_lag7", "tdry_std_lag7",
                      "csky_ghi_mean_lag7", "csky_ghi_std_lag7",
                      "gh_mean_lag30", "gh_std_lag30", "tdry_mean_lag30", "tdry_std_lag30",
                      "csky_ghi_mean_lag30", "csky_ghi_std_lag30",
                      "month", "week", "day", "day_of_week"]


class Arima:

    def __init__(self, data, target, table):
        data.reset_index(inplace=True)
        self.data = data
        self.target = str(target)
        split_loc = int(len(data[target]) * 0.66)
        split_table = table.split("_")
        split_table.pop(0)
        split_table.pop(-1)
        self.table_name = "_".join(split_table)
        self.split_date = data['datetime'][split_loc]
        self.lag_features()
        self.prepare_data()
        self.train_arima()

    def lag_features(self):
        df = self.data
        df.reset_index(drop=True, inplace=True)
        lag_features = ['gh', 'csky_ghi', 'tdry']
        window1 = 3
        window2 = 7
        window3 = 30

        df_rolled_3d = df[lag_features].rolling(window=window1, min_periods=0)
        df_rolled_7d = df[lag_features].rolling(window=window2, min_periods=0)
        df_rolled_30d = df[lag_features].rolling(window=window3, min_periods=0)

        df_mean_3d = df_rolled_3d.mean().shift(1).reset_index().astype(np.float32)
        df_mean_7d = df_rolled_7d.mean().shift(1).reset_index().astype(np.float32)
        df_mean_30d = df_rolled_30d.mean().shift(1).reset_index().astype(np.float32)

        df_std_3d = df_rolled_3d.std().shift(1).reset_index().astype(np.float32)
        df_std_7d = df_rolled_7d.std().shift(1).reset_index().astype(np.float32)
        df_std_30d = df_rolled_30d.std().shift(1).reset_index().astype(np.float32)

        for feature in lag_features:
            df[f"{feature}_mean_lag{window1}"] = df_mean_3d[feature]
            df[f"{feature}_mean_lag{window2}"] = df_mean_7d[feature]
            df[f"{feature}_mean_lag{window3}"] = df_mean_30d[feature]

            df[f"{feature}_std_lag{window1}"] = df_std_3d[feature]
            df[f"{feature}_std_lag{window2}"] = df_std_7d[feature]
            df[f"{feature}_std_lag{window3}"] = df_std_30d[feature]

        df.fillna(df.mean(), inplace=True)

        df.set_index("datetime", drop=False, inplace=True)

    def prepare_data(self):
        df = self.data
        date = self.split_date

        df.datetime = pd.to_datetime(df.datetime, format="%Y-%m-%d %H:%M:%S")
        df["month"] = df.datetime.dt.month
        df["week"] = df.datetime.dt.week
        df["day"] = df.datetime.dt.day
        df["day_of_week"] = df.datetime.dt.dayofweek

        df_train = df[df.datetime < date]
        df_valid = df[df.datetime >= date]

        self.df_train = df_train
        self.df_valid = df_valid

    def train_arima(self):
        df_train = self.df_train.copy(deep=False)
        df_valid = self.df_valid.copy(deep=False)
        target = self.target
        date_time = str(datetime.datetime.now().strftime("%d_%m_%Y"))

        model = auto_arima(df_train[target], exogenous=df_train[exogenous_features], trace=True,
                           error_action="ignore", suppress_warnings=True)
        model.fit(df_train[target], exogenous=df_train[exogenous_features])

        forecast = model.predict(n_periods=len(df_valid), exogenous=df_valid[exogenous_features])
        df_valid["Forecast_ARIMAX"] = forecast
        self.df_valid = df_train

        filename_temp = 'temp_arima_model_' + self.target + '_' + date_time + '.pickle'
        # save the model to disk
        self.filename_temp = filename_temp
        with open(filename_temp, 'wb') as filename_temp:
            pickle.dump(model, filename_temp)

    def save_model_localy(self):
        # save the model to disk
        with open(self.filename_temp, 'rb') as filename_temp:
            model = pickle.load(filename_temp)
        date_time = str(datetime.datetime.now().strftime("%d_%m_%Y"))
        filename_arima = 'arima_model_' + self.target + '_' + date_time + '.pickle'
        if os.path.exists(filename_arima):
            os.remove(filename_arima)
        self.filename_arima = filename_arima
        with open(filename_arima, 'wb') as filename_arima:
            pickle.dump(model, filename_arima)

    # def save_model_s3(self):
    #     upload_s3(self.filename_arima)

    def validation_arima(self):
        target = self.target
        print("RMSE of Auto ARIMAX:", np.sqrt(mean_squared_error(self.df_valid[target], self.df_valid.Forecast_ARIMAX)))
        print("\nMAE of Auto ARIMAX:", mean_absolute_error(self.df_valid[target], self.df_valid.Forecast_ARIMAX))

    def get_model_arima(self):
        with open(self.filename_arima, 'rb') as filename_arima:
            model = pickle.load(filename_arima)
        return model


def main():
    table = 'tsdb_cams_mera_cleaned'
    data = read_data(table)
    target = data.columns[-1]
    arima_prediction = Arima(data, target, table)
    arima_prediction.save_model_localy()
    table = 'tsdb_sma_weatherbit_cleaned'
    data = read_data(table)
    target = data.columns[-1]
    arima_prediction = Arima(data, target, table)
    arima_prediction.save_model_localy()


if __name__ == "__main__":
    main()
