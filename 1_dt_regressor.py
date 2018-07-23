import pandas as pd
import numpy as np
from dr import SQLConnector
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import model_selection
s = SQLConnector(host="localhost",pwd="r0b0t161",db="db_crypto",user="robot")
query = "SELECT * FROM `prices1m` LIMIT 61458"
#data_p.drop(['id', 'ticker', 'mts', 'updated_at'], axis = 1)
data_p = pd.read_csv("prices1m.csv")
data_p.columns = ['id', 'ticker', 'mts', 'open', 'close', 'high', 'low', 'volume', 'updated_at']
df = pd.DataFrame()
df['open'] = data_p['open']
df['close'] = data_p['close']
df['high'] = data_p['high']
df['low'] = data_p['low']
df['volume'] = data_p['volume']
data_r = pd.read_csv("trendhistory_2018_06_13.csv", nrows = 50044)

X = np.asarray(df)
y = np.asarray(df['close'])
#print(X.shape[0])
#print(y.shape[0])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
kfold = model_selection.KFold(n_splits = 10 , random_state = 7)
reg = tree.DecisionTreeRegressor()

reg.fit(X_train,y_train)
y_true, y_pred = y_test, reg.predict(X_test)
results = model_selection.cross_val_score(reg, X, y ,cv = kfold, scoring = 'neg_mean_squared_error')
print('Score regression: {}'.format(reg.score(X_train, y_train)))
print('Mean squared error {}'.format(results.std()))
print('Predicted regression:{}'.format(reg.predict(X_test)))
