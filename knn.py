import pandas as pd
import numpy as np
from dr import SQLConnector
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import sys
s = SQLConnector(host="localhost",pwd="r0b0t161",db="db_crypto",user="robot")
query = "SELECT * FROM `prices1m` ORDER BY `updated_at` DESC LIMIT 61458 "
data_p = pd.read_csv("prices1m.csv")
data_p.columns = ['id', 'ticker', 'mts', 'open', 'close', 'high', 'low', 'volume', 'updated_at']
#print(data_p['open'])
data_r = pd.read_csv("trendhistory_2018_06_13.csv", nrows = 50045)
df = pd.DataFrame()
df['open'] = data_p['open']
df['close'] = data_p['close']
df['high'] = data_p['high']
df['low'] = data_p['low']
df = df.dropna()
df.iloc[:,:4]
#print(df.head())
df['S_10'] = df['close'].rolling(window=10).mean()
df['Corr'] = df['close'].rolling(window=10).corr(df['S_10'])
#df['RSI'] = ta.RSI(np.array(df['close']), timeperiod =10)
df['Open-Close'] = df['open'] - df['close'].shift(1)
df['Open-Open'] = df['open'] - df['open'].shift(1)
df = df.dropna()
X = df.iloc[:,:9]
y = np.where (df['close'].shift(-1) > df['close'],1,-1)
split = int(0.7*len(df))

X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
model = KNeighborsClassifier(n_neighbors = 7)
model.fit(X,y)
print('Score: {}'.format(model.score(X,y)))
print('Predicted :{}'.format(model.predict(X_test)))
