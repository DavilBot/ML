import pandas as pd
import numpy as np
from dr import SQLConnector
import matplotlib.pyplot as plt
from sklearn import linear_model
import math
import sys

s = SQLConnector(host="localhost",pwd="r0b0t161",db="db_crypto",user="robot")
query = "SELECT * FROM `prices1m` LIMIT 61458"
data_p = pd.read_csv("prices1m.csv")
data_p.columns = ['id', 'ticker', 'mts', 'open', 'close', 'high', 'low', 'volume', 'updated_at']
#data_p.drop(['id', 'ticker', 'mts', 'updated_at'], axis = 1)
df = pd.DataFrame()
df['open'] = data_p['open']
df['close'] = data_p['close']
df['high'] = data_p['high']
df['low'] = data_p['low']
df['volume'] = data_p['volume']
#df['volume'] = data_p['volume']
data_r = pd.read_csv("trendhistory_2018_06_13.csv", nrows = 49543)
df['h_l'] = df['high']-df['low']
df['c_o'] = df['close']-df['open']
df = df[['h_l', 'c_o', 'close', 'volume']]
df.fillna(-989898, inplace = True)
df['y'] = df['close'].shift(-int(math.ceil(0.01*len(df))))
#print(df.head())
#sys.exit()
X = np.array(df.drop(['y'], 1))
X_l = X[-int(math.ceil(0.01*len(df))):]
X = X[:-int(math.ceil(0.01*len(df)))]
df.dropna(inplace = True)
y = np.array(df['y'])
#print(X.shape[0])
#print(y.shape[0])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

reg = linear_model.LinearRegression(n_jobs = -1)
reg.fit(X_train,y_train)
print('Coefficients: \n', reg.coef_)
print('Score: {}'.format(reg.score(X_test, y_test)))
print('Predicted :{}'.format(reg.predict(X_l)))
