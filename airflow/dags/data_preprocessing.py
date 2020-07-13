import pandas as pd
from airflow.hooks.postgres_hook import PostgresHook
# import numpy as np
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from matplotlib import pyplot
from pandas import read_sql_table


# from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
# from sklearn.preprocessing import StandardScaler
# full documentation: http://www.statsmodels.org/dev/_modules/statsmodels/stats/outliers_influence.html#variance_inflation_factor
# from statsmodels.stats.outliers_influence import variance_inflation_factor

def process():
    SQLALCHEMY_DATABASE_URI = 'postgresql://{db_user}:{db_password}@{db_host}:5432/{db_name}'.format(
                db_host = '34.69.215.94',
                db_name = 'snpiao_data',
                db_password = 'ucamnafinki',
                db_user = 'snpiao_data',
            )

    PostgresHook(conn_name_attr='')

    energy = read_sql_table('tsdb_fronius_energy',SQLALCHEMY_DATABASE_URI)
    cams = read_sql_table('tsdb_cams', SQLALCHEMY_DATABASE_URI)
    merra = read_sql_table('tsdb_merra', SQLALCHEMY_DATABASE_URI)

    print("ok")

    energy = energy.set_index('datetime').sort_index()
    cams = cams.set_index('datetime').sort_index()
    merra = merra.set_index('datetime').sort_index()

    #aggregate with sum over every hour
    energy = energy.resample('1H').agg('sum')

    print("ok")

    merged = cams.merge(energy, on='datetime')
    merged = merged.merge(merra, on='datetime')
    #merged.fillna(method='ffill', inplace=True)

    #more sophisticated interpolation technique instead of only repeating the last valid sample with ffill
    merged.interpolate(method='linear', inplace=True)

    merged['hour'] = merged.index.hour
    merged['dayofweek'] = merged.index.dayofweek
    merged['consumption'] = merged['FromGenToConsumer'] + merged['FromGridToConsumer']

    #get rid of outlier
    q = merged['consumption'].quantile(0.99)
    merged = merged[merged['consumption'] < q]

    pyplot.plot(merged['consumption'])
    pyplot.show()

    print(merged.shape)
    print(merged)

    X = pd.DataFrame(data=merged, columns=['gh', 'csky_ghi', 'hour', 'dayofweek', 'tdry', 'wspd', 'rainfall'])
    y = pd.DataFrame(data=merged, columns=['consumption'])
    final = pd.DataFrame(data=merged, columns=['gh', 'csky_ghi', 'hour', 'dayofweek', 'tdry', 'wspd', 'rainfall', 'consumption'])

    X.to_csv('clean_data.csv', encoding='utf-8')
    y.to_csv('clean_targets.csv', encoding='utf-8')
    final.to_csv('merged_data.csv', encoding='utf-8')

    print(X.shape)
    print(X)
    #X.isnull().sum()
    print(X.isnull().values.any())
    print(y.shape)
    print(y)
    print(y.isnull().sum())

    # To make this as easy as possible to use, we declare a variable where we put
    # all features where we want to check for multicollinearity
    # since our categorical data is not yet preprocessed, we will only take the numerical ones
    # variables = X[X.columns]
    #
    # # we create a new data frame which will include all the VIFs
    # # note that each variable has its own variance inflation factor as this measure is variable specific (not model specific)
    # vif = pd.DataFrame()
    #
    # # here we make use of the variance_inflation_factor, which will basically output the respective VIFs
    # vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
    # # Finally, I like to include names so it is easier to explore the result
    # vif["Features"] = variables.columns
    #
    # print(vif)
    #The features: gh and csky_ghi (Clear sky global irradiation on horizontal plane at ground level)\
    # have the highest VIF, so we should drop them to improve our regression model.

    #x = StandardScaler().fit_transform(X)
