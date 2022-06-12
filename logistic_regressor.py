from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
iris = datasets.load_iris()
# print(list(iris.keys()))
# # print(list(iris['data']))
# # print(list(iris['target']))
# print(list(iris['DESCR']))
X = iris["data"][:, 3:]
y = (iris["target"] == 2).astype(np.int64)

# train a logistic regression modei
clf = LogisticRegression()
clf.fit(X,y)
example= clf.predict(([[1.6]]))
print(example)

# using matplot lib for plotting
# gives 100 points bwt 0aand3 
X_new = np.linspace(0,3,100).reshape(-1,1) #-1,1 many rows 1 coloumn
y_prob = clf.predict_proba(X_new)
plt.plot(X_new, y_prob[:,1], "g-", label="virginica")
plt.show()

