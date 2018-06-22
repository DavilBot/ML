import time
import pandas as pd
import numpy as np
from dr import SQLConnector
import matplotlib.pyplot as plt
from sklearn import linear_model
#import talib as ta
s = SQLConnector(host="localhost",pwd="r0b0t161",db="db_crypto",user="robot")
query = "SELECT * FROM `prices1m` LIMIT 61458"
data_p = pd.DataFrame(s.exec_sql(query))
#data_p.drop(['id', 'ticker', 'mts', 'updated_at'], axis = 1)
df = pd.DataFrame()
df['open'] = data_p['open']
df['close'] = data_p['close']
df['high'] = data_p['high']
df['low'] = data_p['low']
df['volume'] = data_p['volume']
data_r = pd.read_csv("trendhistory_2018_06_13.csv")#, nrows = 56040)
df = df.dropna()
df.iloc[:,:4]
print(df.head())
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
model = linear_model.LogisticRegression()

model = model.fit (X_train,y_train)
model.score(X_train, y_train)
#X = data_r['percent_number'].iloc[:,1:]
#y = (data_r['percent_number']!=0)*1
#print(X.shape[0])
#print(y.shape[0])
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

