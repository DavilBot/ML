import time
import pandas as pd
import numpy as np
from dr import SQLConnector
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix
import sys
from timeit import default_timer as timer
start_time = timer()
#import talib as ta
s = SQLConnector(host="localhost",pwd="r0b0t161",db="db_crypto",user="robot")
query = "SELECT * FROM `prices1m` ORDER BY `updated_at` DESC LIMIT 61458"
data_p = pd.read_csv("prices1m.csv")
data_p.columns = ['id', 'ticker', 'mts', 'open', 'close', 'high', 'low', 'volume', 'updated_at']
#data_p.drop(['id', 'ticker', 'mts', 'updated_at'], axis = 1)
df = pd.DataFrame()
#df['updated_at'] = data_p['updated_at']
df['open'] = data_p['open']
df['close'] = data_p['close']
df['high'] = data_p['high']
df['low'] = data_p['low']
#df['volume'] = data_p['volume']
data_r = pd.read_csv("trendhistory_2018_06_13.csv", nrows = 50045)
df = df.dropna()
df = df.iloc[:,:4]
X = df#.iloc[:,:9]
y = np.where (df['close'].shift(-1) > df['close'],1,-1)
X1 = np.array(df)
y1 = np.array(df['close'])
kfold = model_selection.KFold(n_splits = 10, random_state = 7)
split = int(0.7*len(df))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
X1_train, X1_test, y1_train, y1_test = X1[:split], X1[split:], y1[:split], y1[split:]
model = RandomForestClassifier(n_jobs = -1, random_state = 0)
reg = RandomForestRegressor(n_jobs = -1, random_state = 0)
model = model.fit(X_train,y_train)
reg = reg.fit(X1_train, y1_train)
y_true, y_pred = y_test, model.predict(X_test)
results = model_selection.cross_val_score(reg, X1, y1, cv = kfold, scoring = 'neg_mean_squared_error')
print('Classification report {}'.format(classification_report(y_true, y_pred)))
print('Confusion matrix {}'.format(confusion_matrix(y_true, y_pred)))
print('Mean_squarred_error {}'.format(results.std()))
#print(model.score(X_train, y_train))
#print(model.predict(X_test))
#print(str(timer()-start_time) + "TIME")
