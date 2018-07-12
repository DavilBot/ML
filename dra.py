import pandas as pd
import numpy as np
from dr import SQLConnector
import matplotlib.pyplot as plt
from sklearn import decomposition

s = SQLConnector(host="localhost",pwd="r0b0t161",db="db_crypto",user="robot")
query = "SELECT * FROM `prices1m` LIMIT 61458"
data_p = pd.read_csv("prices1m.csv")
data_p.columns = ['id', 'ticker', 'mts', 'open', 'close', 'high', 'low', 'volume', 'updated_at']

df = pd.DataFrame()
df['open'] = data_p['open']
df['close'] = data_p['close']
df['high'] = data_p['high']
df['low'] = data_p['low']
data_r = pd.read_csv("trendhistory_2018_06_13.csv", nrows = 50044)

X = np.asarray(df)
y = np.asarray(data_r['percent_number'])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

reg = linear_model.LinearRegression()
reg.fit(X_train,y_train)
print('Coefficients: \n', reg.coef_)
print('Score: {}'.format(reg.score(X_train, y_train)))
print('Predicted :{}'.format(reg.predict(X_test)))
