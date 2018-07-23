import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
data_p = pd.read_csv("prices1m.csv")
data_p.columns = ['id', 'ticker', 'mts', 'open', 'close', 'high', 'low', 'volume', 'updated_at']
df = pd.DataFrame()
df['open'] = data_p['open']
df['close'] = data_p['close']
df['high'] = data_p['high']
df['low'] = data_p['low']
df['volume'] = data_p['volume']
X = np.array(df.drop(['close'], 1))
y = np.where (df['close'].shift(-1) > df['close'],1,-1)#np.array(df['close'])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model = GradientBoostingClassifier(random_state = 4)
model.fit(X_train, y_train)
pred = model.predict(X_train)
pred_prob = model.predict_proba(X_train)[:,1]
cv_score = cross_validation.cross_val_score(model, X_train, y_train, cv = 5, scoring='roc_auc')
print(metrics.roc_auc_score(y_train, pred_prob))
print(cv_score)
print(metrics.accuracy_score(y_train, pred))
#def modelfit(alg, dtrain, predictors, performCV=True, printFeatureImportance=True, cv_folds=5):
#    #Fit the algorithm on the data
#    alg.fit(dtrain[predictors], dtrain['close'])
#    #Predict training set:
#    dtrain_predictions = alg.predict(dtrain[predictors])
#    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
#    #Perform cross-validation:
#    if performCV:
#        cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], dtrain['Disbursed'], cv=cv_folds, scoring='roc_auc')
#    #Print model report:
#        print("\nModel Report")
#        print("Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions))
#        print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob))
#    if performCV:
#        print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
#    #Print Feature Importance:
#    if printFeatureImportance:
#        feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
#        feat_imp.plot(kind='bar', title='Feature Importances')
#        plt.ylabel('Feature Importance Score')
