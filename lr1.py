from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import AdaBoostRegressor
models = [
  LinearRegression(), 
  SVR(), 
  KNeighborsRegressor(n_neighbors=2), 
  DecisionTreeRegressor(),
  GradientBoostingRegressor(),
  GaussianProcessRegressor(),
  PLSRegression(),
  AdaBoostRegressor()
]
# Dataset
train_X = [[5,3,2],[9,2,4],[8,6,3],[5,4,5]]
train_Y = [151022,183652,482466,202541]
# Train each model individually
for model in models:
  model.fit(train_X, train_Y)
  acc = model.score(train_X, train_Y)
  if acc == 1:
    print (model)
    print ("Accuracy: %d" % acc)
    print ("7+2+5 = %d" % model.predict([[7,2,5]]))
