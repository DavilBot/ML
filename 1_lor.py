import time
import pandas as pd
import numpy as np
from dr import SQLConnector
from sklearn import linear_model, model_selection
from sklearn.metrics import classification_report, confusion_matrix
#import talib as ta
s = SQLConnector(host="localhost",pwd="r0b0t161",db="db_crypto",user="robot")
query = "SELECT * FROM `prices1m` ORDER BY `updated_at` DESC LIMIT 61458"
#data_p = pd.DataFrame(s.exec_sql(query))
data_p = pd.read_csv("prices1m.csv")
data_p.columns = ['id', 'ticker', 'mts', 'open', 'close', 'high', 'low', 'volume', 'updated_at']
#data_p.drop(['id', 'ticker', 'mts', 'updated_at'], axis = 1)
df = pd.DataFrame()
df['open'] = data_p['open']
df['close'] = data_p['close']
df['high'] = data_p['high']
df['low'] = data_p['low']
#df['volume'] = data_p['volume']
data_r = pd.read_csv("trendhistory_2018_06_13.csv", nrows = 50044)
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
y_true, y_pred = y_test, model.predict(X_test)
kfold = model_selection.KFold(n_splits=10, random_state=7)
scoring = "roc_auc"
results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
print(results.mean())
print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
print(model.score(X_train, y_train))

