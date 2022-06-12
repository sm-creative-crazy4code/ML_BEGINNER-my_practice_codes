from sklearn import datasets, linear_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


# importing dataset for diabetes
diabetes=datasets.load_diabetes()

# prints all the data of the key at index 2 in colounnsssss
diabetes_X= diabetes.data
# print(diabetes_X)

# 'data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'
diabetes_X_train = diabetes_X[:-30] #taking last 3 datapoints for traning

diabetes_X_test = diabetes_X[ :-20] # taking first 20 for training

diabetes_y_train = diabetes.target[:-30] #taking last 3 datapoints for traning

diabetes_y_test = diabetes.target[ :-20]




model=linear_model.LinearRegression()

model.fit(diabetes_X_train,diabetes_y_train)#model has been created

#Now we are testing the model to 
diabetes_y_predict = model.predict(diabetes_X_test,)
# plt.scatter(diabetes_X_test,diabetes_y_test)
# plt.plot(diabetes_X_test,diabetes_y_predict)
# plt.show()


print("MEAN SQUARED ERROR:",mean_squared_error(diabetes_y_test,diabetes_y_predict))
print("weights:", model.coef_ )
print("intercepts:", model.intercept_)

# MEAN SQUARED ERROR: 3955.2731368433965
# weights: [941.43097333]
# intercepts: 153.39713623331644


