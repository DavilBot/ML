




import pandas as pd
import numpy as np
from dr import SQLConnector
import matplotlib.pyplot as plt
from sklearn import linear_model

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

X = np.asarray(df)
y = np.asarray(data_r['percent_number'])
#print(X.shape[0])
#print(y.shape[0])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

reg = linear_model.LinearRegression()
reg.fit(X_train,y_train)
print('Coefficients: \n', reg.coef_)
print('Score: {}'.format(reg.score(X_train, y_train)))
print('Predicted :{}'.format(reg.predict(X_test)))
