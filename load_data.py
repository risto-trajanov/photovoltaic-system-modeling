import pandas as pd
from pandas import read_sql_table
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SQLALCHEMY_DATABASE_URI = 'postgresql://{db_user}:{db_password}@{db_host}:5432/{db_name}'.format(
            db_host = '34.69.215.94',
            db_name = 'snpiao_data',
            db_password = 'xxx',
            db_user = 'snpiao_data',
        )

params = {
    "learning_rate":     [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    "min_samples_split": np.linspace(0.1, 0.5, 3),
    "min_samples_leaf":  np.linspace(0.1, 0.5, 3),
    "max_depth":         [3, 5, 8],
    "max_features":      ["log2", "sqrt"],
    "criterion":         ["friedman_mse", "mae"],
    "subsample":         [0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
    "n_estimators":      [5, 10, 25]
        }


params = {'criterion': 'friedman_mse', 'learning_rate': 0.15, 'max_depth': 8, 'max_features': 'log2', 'min_samples_leaf': 0.1, 'min_samples_split': 0.5, 'n_estimators': 5, 'subsample': 0.9}
energy = read_sql_table('tsdb_fronius_energy',SQLALCHEMY_DATABASE_URI)
cams = read_sql_table('tsdb_cams', SQLALCHEMY_DATABASE_URI)
merra = read_sql_table('tsdb_merra', SQLALCHEMY_DATABASE_URI)

energy = energy.set_index('datetime').sort_index()
cams = cams.set_index('datetime').sort_index()
merra = merra.set_index('datetime').sort_index()
energy = energy.resample('1H').agg('mean')

merged = cams.merge(energy, on='datetime')
merged = merged.merge(merra, on='datetime')
merged.fillna(method='ffill', inplace=True)
merged['hour'] = merged.index.hour
merged['dayofweek'] = merged.index.dayofweek
merged['consumption'] = merged['FromGenToConsumer'] + merged['FromGridToConsumer']
pyplot.plot(merged['consumption'])
pyplot.show()
X = pd.DataFrame(data=merged, columns=['gh', 'csky_ghi', 'hour', 'dayofweek', 'tdry', 'wspd', 'rainfall'])
y = pd.DataFrame(data=merged, columns=['consumption'])

x = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

clf = GradientBoostingRegressor(**params)

score = cross_val_score(clf, X, y, cv=5, scoring='neg_mean_absolute_error')

#model = GridSearchCV(clf, param_grid=params, cv=3, n_jobs=-1)
#model.fit(principalDf, y.values.ravel())

print('ok')