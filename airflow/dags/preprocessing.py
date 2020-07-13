import pandas as pd
import numpy as np
from utils import persist_data
from utils import read_data


class Transform:
    def __init__(self, data):
        data.reset_index(inplace=True)
        self.data = data

    def get_data(self):
        lag = self.lag_features()
        ret_data = self.prepare_data(lag)
        return ret_data

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
        return df

    def prepare_data(self, data):
        df = data

        df.datetime = pd.to_datetime(df.datetime, format="%Y-%m-%d %H:%M:%S")
        df["month"] = df.datetime.dt.month
        df["week"] = df.datetime.dt.week
        df["day"] = df.datetime.dt.day
        df["day_of_week"] = df.datetime.dt.dayofweek

        return df


class Preprocessing:

    def __init__(self, features, table):
        self.features = features
        self.table = table

    def mers_cams_data(self):
        target = 'consumption'

        energy = read_data('tsdb_fronius_energy')
        cams = read_data('tsdb_cams')
        merra = read_data('tsdb_merra')

        energy.index.name = 'datetime'
        cams.index.name = 'datetime'
        merra.index.name = 'datetime'
        energy = energy.resample('1H').agg('mean')

        merged = cams.merge(energy, on='datetime')
        merged = merged.merge(merra, on='datetime')
        merged['consumption'] = merged['FromGenToConsumer'] + merged['FromGridToConsumer']

        q = merged[target].quantile(0.99)
        merged = merged[merged[target] < q]
        df = self.to_DataFrame(merged, self.features, target)
        persist_data(self.table, df)

    def to_DataFrame(self, data, features, target):
        features_model = features
        features_model.append(target)
        return pd.DataFrame(data=data, columns=features_model)

    def weatherbit(self):
        target = 'energy'
        features_data = read_data('tsdb_weatherbit_cleaned')
        target_data = read_data('tsdb_sma_energy')

        features_data.index.name = 'datetime'
        target_data.index.name = 'datetime'

        merged = features_data.merge(target_data, on='datetime')

        q = merged[target].quantile(0.99)
        merged = merged[merged[target] < q]

        df = self.to_DataFrame(merged, self.features, target)
        persist_data(self.table, df)


def main():
    features = ['gh', 'csky_ghi', 'tdry']
    table = 'tsdb_cams_mera_cleaned'
    preprocess = Preprocessing(features, table)
    preprocess.mers_cams_data()
    features = ['gh', 'csky_ghi', 'tdry']
    table = 'tsdb_sma_weatherbit_cleaned'
    preprocess = Preprocessing(features, table)
    preprocess.weatherbit()


if __name__ == "__main__":
    main()
