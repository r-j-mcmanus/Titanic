#
# random forest space 
# support vector machine
#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Feature_Engineering import Feature_Engineering

df = pd.read_csv('train.csv')
df = Feature_Engineering(df)

print(df.info())
print(df.head())

#from histograms import make_histograms
#from scatter_plots import make_scatter_plots
#make_histograms(df)
#make_scatter_plots(df)

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from hyperparam_fit import *

df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

scaler = StandardScaler()
df[['Age']] = scaler.fit_transform(df[['Age']])
df[['Fare']] = scaler.fit_transform(df[['Fare']])

y = df['Survived'].values
X = df.drop('Survived', axis = 1).values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,random_state=42)


###

print('knn')
knn = knn_best_fit(X_train, y_train)
knn = knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
knnScore = knn.score(X_test, y_test)
print(knnScore)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print('')

###

print('Logistic Regression')
logreg = LogisticRegression_best_fit(X_train, y_train)
logreg = logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
logregScore = logreg.score(X_test, y_test)
print(logregScore)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print('')

###

print('Tree')
tree = tree_best_fit(X_train, y_train)
tree = tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
treeScore = tree.score(X_test, y_test)
print(treeScore)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print('')


###

print('SVC')
svc = SVC_best_fit(X_train, y_train)
svc = svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
SVCScore = svc.score(X_test, y_test)
print(SVCScore)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print('')

####

from sklearn.ensemble import VotingClassifier
#we have multiple clasifiers which will have internal bias
#adverage them to get a better estimator

score_total = knnScore + logregScore + treeScore + SVCScore

votingClassifier = VotingClassifier(
	estimators = [('knn', knn),('logreg', logreg),('tree',tree),('svc',svc)],
	voting = 'soft',
	weights = [knnScore/score_total,logregScore/score_total,treeScore/score_total,SVCScore/score_total]
	)

votingClassifier.fit(X_train,y_train)
y_pred = votingClassifier.predict(X_test)
print('votingClassifier')
print(votingClassifier.score(X_test, y_test))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#retrain all on the full dataset

votingClassifier.fit(X,y)

##

df_test = pd.read_csv('test.csv')
df_test['Survived'] = 1

df_test = Feature_Engineering(df_test)

df_test['Age'] = df_test['Age'].fillna(df_test['Age'].mean())
df_test['Fare'] = df_test['Fare'].fillna(df_test['Fare'].median())
df_test[['Age']] = scaler.fit_transform(df_test[['Age']])
df_test[['Fare']] = scaler.fit_transform(df_test[['Fare']])

X = df_test.drop('Survived', axis = 1).values

predictions = votingClassifier.predict(X)
df_test['predictions'] = predictions

df_test['predictions'].to_csv('titanic_predictions.csv')

