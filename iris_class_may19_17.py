# numerical library
import numpy as np

# the classifiers we want to use
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# import the data set and other useful functions
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# set seed so each run is the same.
# in particular this number was initially chosen so that the
# regions carved out by KMeans (which are randomly assigned labels)
# would agree with the data set labels, (i.e., 0, 1, and 2)
np.random.seed(2)


# load the data
iris = load_iris()
X = iris.data
y = iris.target


# split into training and testing data for supervised learning
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.33,
                                                    random_state=43)


# KMeans classifier
clfKM = KMeans(n_clusters=3)
clfKM.fit(X)
predKM = clfKM.predict(X)


# Naive Bayes classifier
clfNB = GaussianNB()
clfNB.fit(X_train, y_train)
predNB = clfNB.predict(X_test)


# Decision Tree classifier
clfDT = DecisionTreeClassifier(min_samples_split=10)
clfDT.fit(X_train, y_train)
predDT = clfDT.predict(X_test)


# output the accuracies of each respective classifier
print "KMeans accuracy: ", accuracy_score(y, predKM)
print "Naive Bayes accuracy: ", accuracy_score(y_test, predNB)
print "Decision Tree accuracy: ", accuracy_score(y_test, predDT)

