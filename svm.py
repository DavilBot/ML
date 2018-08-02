from pandas_datareader import data as web
import pandas as pd
import numpy as np
from dr import SQLConnector
import matplotlib.pyplot as plt
from sklearn import mixture as mix
import seaborn as sns
import ta
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import sys
import matplotlib.pyplot as plt
%matplotlib inline
sys.exit()
data_p = pd.read_csv("prices1m.csv")
data_p.columns = ['id', 'ticker', 'mts', 'open', 'close', 'high', 'low', 'volume', 'updated_at']
#print(data_p['open'])
data_r = pd.read_csv("trendhistory_2018_06_13.csv", nrows = 50045)
df = pd.DataFrame()
df['open'] = data_p['open']
df['close'] = data_p['close']
df['high'] = data_p['high']
df['low'] = data_p['low']
df['Close'] = df['close'].shift(1)
df['High'] = df['high'].shift(1)
df['Low'] = df['low'].shift(1)
df['RSI'] = RSI(df['Close'], 10)
df['SAR'] = ta.SAR(np.array(df['High']), np.array(df["Low"]),0.2,0.2)
df['ADX'] = ta.ADX(np.array(df['High']), np.array(df['Low']),np.array(df['Close']), timeperiod = 10)
df['Return'] = np.log(df['open']/df['open'].shift(1))
print(df.head())
#df = df.dropna()
#df.iloc[:,:4]
df = df.dropna()
X = df.iloc[:,:9]
y = np.where (df['close'].shift(-1) > df['close'],1,-1)
split = int(0.7*len(df))

X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
model = SVC(kernel='linear',gamma=1)
model.fit(X,y)
print('Score: {}'.format(model.score(X_train, y_train)))
print('Predicted :{}'.format(model.predict(X_test)))
def RSI(series, period):
     delta = series.diff().dropna()
     u = delta * 0
     d = u.copy()
     u[delta > 0] = delta[delta > 0]
     d[delta < 0] = -delta[delta < 0]
     u[u.index[period-1]] = np.mean( u[:period] ) #first value is sum of avg gains
     u = u.drop(u.index[:(period-1)])
     d[d.index[period-1]] = np.mean( d[:period] ) #first value is sum of avg losses
     d = d.drop(d.index[:(period-1)])
     rs = pd.stats.moments.ewma(u, com=period-1, adjust=False) / \
     pd.stats.moments.ewma(d, com=period-1, adjust=False)
     return 100 - 100 / (1 + rs)
