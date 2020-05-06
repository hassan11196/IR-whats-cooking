import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




data = pd.read_csv('./data/small_data.csv',na_values='-')
data.fillna(0,inplace=True)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

target = 'cuisine'
columns = data.columns.tolist()
columns = [c for c in columns if c not in ['cuisine']]
data = data[columns]
print(data.shape)
print(data.columns)
train = data.sample(frac=0.8,random_state=1)
test = data.loc[~data.index.isin(train.index)]


LR = LogisticRegression()
KNN = KNeighborsClassifier(n_neighbors=10)
svm = SVC()

# LR.fit(train[columns],train[target])
# pred = LR.predict(test[columns])

# print("Logistic Regression")
# print(accuracy_score(test[target],pred))

# KNN.fit(train[columns],train[target])
# pred = KNN.predict(test[columns])

# print("K neighbors")
# print(accuracy_score(test[target],pred))

# # svm.fit(train[columns],train[target])
# # pred = svm.predict(test[columns])


# gnb = GaussianNB()
# gnb.fit(train[columns],train[target])
# pred= gnb.predict(test[columns])
# print("GNB")
# print(accuracy_score(test[target],pred))
