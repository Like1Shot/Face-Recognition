import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

data = pd.read_csv("output.csv")

X = data.drop(['4096', '4097'], axis=1)
y = data['4097']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)


# # Logistic Regression
# lr = LogisticRegression(penalty='elasticnet', dual=False, tol=0.0001, C=10.0, fit_intercept=True, intercept_scaling=1, class_weight='balanced', random_state=None, solver='saga', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=0.5)
# lr.fit(X_train, y_train)
# predicted = lr.predict(X_test)
# score = lr.score(X_test, y_test)
# print('Accuracy of Logistic Regression: ', score)
# f1_score = f1_score(np.array(y_test), np.array(predicted), average=macro)
# print('F1 score of Logistic Regression: ', f1_score)

# Stochastic Gradient Descendent
sc = SGDClassifier(loss='log', max_iter=100)
sc.fit(X_train, y_train)
predicted = sc.predict(X_test)
accuracy = sc.score(X_test, y_test)
print('Accuracy of Stochastic Gradient Descendent: ', accuracy)
f1 = f1_score(np.array(y_test), np.array(predicted), average="macro")
print('F1 score of Stochastic Gradient Descendent: ', f1)

# Random Forest
rf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rf.fit(X_train, y_train)
predicted = rf.predict(X_test)
accuracy = rf.score(X_test, y_test)
print('Accuracy of Random Forest: ', accuracy)
f1 = f1_score(np.array(y_test), np.array(predicted), average="macro")
print('F1 score of Random Forest: ', f1)

# AdaBoost
ada = AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42)
ada.fit(X_train, y_train)
predicted = ada.predict(X_test)
accuracy = ada.score(X_test, y_test)
print('Accuracy of AdaBoost: ', accuracy)
f1 = f1_score(np.array(y_test), np.array(predicted), average="macro")
print('F1 score of AdaBoost: ', f1)

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', n_jobs=-1)
knn.fit(X_train, y_train)
predicted = knn.predict(X_test)
accuracy = knn.score(X_test, y_test)
print('Accuracy of K-Nearest Neighbors: ', accuracy)
f1 = f1_score(np.array(y_test), np.array(predicted), average="macro")
print('F1 score of K-Nearest Neighbors: ', f1)


# Multilayer-Perceptron
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
mlp.fit(X_train, y_train)
accuracy = mlp.score(X_test, y_test)
print('Accuracy of K-Nearest Neighbors: ', accuracy)
f1 = f1_score(np.array(y_test), np.array(predicted), average="macro")
print('F1 score of K-Nearest Neighbors: ', f1)




# features = pd.read_csv("")
# label = pd.read_csv("")

# X = features[]
# y = label

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1, stratify=y)


# knn = KNeighborsClassifier(n_neighbors = 5)
# knn.fit(X, y)
# score_knn.score(X_test, y_test)
# f1_knn = sklearn.metrics.f1_score(y_true, t_pred, average="macro")

# mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
# mlp.fit(X, y)
# core_mlp = score(X_test, y_test)
# f1_mlp = sklearn.metrics.f1_score(y_true, t_pred, average="macro")
