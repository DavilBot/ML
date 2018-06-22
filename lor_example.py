from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# load the iris datasets
dataset = datasets.load_iris()
# create a base classifier used to evaluate a subset of attributes
model = LogisticRegression()
# create the RFE model and select 3 attributes

model.fit(dataset.data, dataset.target)
# display the relative importance of each attribute
print(model.score(dataset.data, dataset.target))
print(model.predict(dataset.data))
