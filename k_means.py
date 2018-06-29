import pandas as pd
import numpy as np
from dr import SQLConnector
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sys
s = SQLConnector(host="localhost",pwd="r0b0t161",db="db_crypto",user="robot")
query = "SELECT * FROM `prices1m` ORDER BY `updated_at` DESC LIMIT 61458 "
data_p = pd.read_csv("prices1m.csv")
data_p.columns = ['id', 'ticker', 'mts', 'open', 'close', 'high', 'low', 'volume', 'updated_at']
#print(data_p['open'])
data_r = pd.read_csv("trendhistory_2018_06_13.csv", nrows = 50045)
o = data_p['open'].values
c = data_p['close'].values
h = data_p['high'].values
l = data_p['low'].values
#print(df.head())

X = np.array(list(zip(o, c, h, l)))
#y = np.where (df['close'].shift(-1) > df['close'],1,-1)
#split = int(0.7*len(df))

#X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
model = KMeans(n_clusters = 4, random_state = 0  )
model.fit(X)
print('Predicted :{}'.format(model.predict(X)))
print('Cluster centers :{}'.format(model.cluster_centers_))
