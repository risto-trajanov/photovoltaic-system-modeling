import datetime
import os
import pickle
from preprocessing import Transform
import numpy as np
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils import read_data
from utils import upload_s3
import pandas as pd
pd.options.mode.chained_assignment = None

# from utils import upload_s3

exogenous_features = ["gh_mean_lag3", "gh_std_lag3", "tdry_mean_lag3", "tdry_std_lag3", "csky_ghi_mean_lag3",
                      "csky_ghi_std_lag3",
                      "gh_mean_lag7", "gh_std_lag7", "tdry_mean_lag7", "tdry_std_lag7",
                      "csky_ghi_mean_lag7", "csky_ghi_std_lag7",
                      "gh_mean_lag30", "gh_std_lag30", "tdry_mean_lag30", "tdry_std_lag30",
                      "csky_ghi_mean_lag30", "csky_ghi_std_lag30",
                      "month", "week", "day", "day_of_week"]


class Arima:

    def __init__(self, data, target, table=False):
        data.reset_index(inplace=True)
        self.data = data
        self.target = str(target)
        split_loc = int(len(data[target]) * 0.66)
        split_table = table.split("_")
        split_table.pop(0)
        split_table.pop(-1)
        self.table_name = "_".join(split_table)
        self.split_date = data['datetime'][split_loc]
        transform = Transform(self.data)
        self.data = transform.get_data()
        self.train_arima()

    def train_arima(self):
        df = self.data
        date = self.split_date

        df_train = df[df.datetime < date]
        df_valid = df[df.datetime >= date]

        target = self.target
        date_time = str(datetime.datetime.now().strftime("%d_%m_%Y"))

        model = auto_arima(df_train[target], exogenous=df_train[exogenous_features], trace=True,
                           error_action="ignore", suppress_warnings=True)
        model.fit(df_train[target], exogenous=df_train[exogenous_features])

        forecast = model.predict(n_periods=len(df_valid), exogenous=df_valid[exogenous_features])
        df_valid["Forecast_ARIMAX"] = forecast
        self.df_valid = df_valid

        filename_temp = 'temp_arima_model_' + self.target + '_' + date_time + '.pickle'
        # save the model to disk
        self.filename_temp = filename_temp
        with open("Models/" + filename_temp, 'wb') as filename_temp:
            pickle.dump(model, filename_temp)

    def save_model_localy(self):
        # save the model to disk
        with open("Models/" + self.filename_temp, 'rb') as filename_temp:
            model = pickle.load(filename_temp)
        date_time = str(datetime.datetime.now().strftime("%d_%m_%Y"))
        filename_arima = 'arima_model_' + self.target + '_' + date_time + '.pickle'
        if os.path.exists(filename_arima):
            os.remove(filename_arima)
        self.filename_arima = filename_arima
        with open("Models/" + filename_arima, 'wb') as filename_arima:
            pickle.dump(model, filename_arima)

    def save_model_s3(self):
        upload_s3(self.filename_arima)

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
    arima_prediction.save_model_s3()
    table = 'tsdb_sma_weatherbit_cleaned'
    data = read_data(table)
    target = data.columns[-1]
    arima_prediction = Arima(data, target, table)
    arima_prediction.save_model_localy()
    arima_prediction.save_model_s3()


if __name__ == "__main__":
    main()
