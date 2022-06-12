from pyexpat import features
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier


iris = datasets.load_iris()


features= iris.data
labels = iris.target
# print(iris.DESCR)
# prebuild data set
# print(features[0],labels[0])
#LABELS   0  - Iris-Setosa
 #    1 - Iris-Versicolour
 #     2- Iris-Virginica

#  TRAINING THE CLASSIFIER
clf = KNeighborsClassifier()
clf.fit(features,labels) # feeding data to classifier
preds = clf.predict([[31,1,1,1]])
print(preds)
# output is stosa

