import pandas as pd
import numpy as np
#import seaborn as sns
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

data = pd.read_csv("output_128.csv")

X = data.drop(['16384', '16385'], axis=1)
y = data['16385']

print(X.head(10))
print(y.head(10))
'''
X = data.drop(['16384', '16385'], axis=1)
y = data['16385']

'''

X_train, X_test, y_train, y_test = train_test_split(X, y
	, test_size = 0.2, random_state=0)

# # Logistic Regression
# lr = LogisticRegression(penalty='elasticnet', dual=False, tol=0.0001, C=10.0, fit_intercept=True, intercept_scaling=1, class_weight='balanced', random_state=None, solver='saga', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=0.5)
# lr.fit(X_train, y_train)
# predicted = lr.predict(X_test)
# score = lr.score(X_test, y_test)
# print('Accuracy of Logistic Regression: ', score)
# f1_score = f1_score(np.array(y_test), np.array(predicted), average=macro)
# print('F1 score of Logistic Regression: ', f1_score)

def runSGD(loss_func, penalty_func, iteration):
	# Stochastic Gradient Descendent
	# loss --> log, modified_huber, squared_hinge, perceptron, hinge(default)
	# penalty --> 'l2(default), l1'
	# alpha set as default 0.0001
	sc = SGDClassifier(loss=loss_func, penalty=penalty_func, max_iter=iteration)
	print("======================= max_iter = " + str(iteration) + " =======================")
	sc.fit(X_train, y_train)
	predicted = sc.predict(X_test)
	accuracy = sc.score(X_test, y_test)
	print('Accuracy of Stochastic Gradient Descendent: ', accuracy)
	f1 = f1_score(np.array(y_test), np.array(predicted), average="macro")
	print('F1 score of Stochastic Gradient Descendent: ', f1)
# change max iteration --> plot

def runRandomForest(n):
	# Random Forest
	rf = RandomForestClassifier(n_estimators=n, n_jobs=-1)
	print("===================== n_estimators = " + str(n) + " =====================")
	rf.fit(X_train, y_train)
	predicted = rf.predict(X_test)
	accuracy = rf.score(X_test, y_test)
	print('Accuracy of Random Forest: ', accuracy)
	f1 = f1_score(np.array(y_test), np.array(predicted), average="macro")
	print('F1 score of Random Forest: ', f1)

def runAdaBoost(n, learn_rate):
	# AdaBoost
	ada = AdaBoostClassifier(n_estimators=n, learning_rate=learn_rate, random_state=42)
	print("=================== n_estimators = " + str(n) + " with learning_rate = " + str(learn_rate) + " ===================")
	ada.fit(X_train, y_train)
	predicted = ada.predict(X_test)
	accuracy = ada.score(X_test, y_test)
	print('Accuracy of AdaBoost: ', accuracy)
	f1 = f1_score(np.array(y_test), np.array(predicted), average="macro")
	print('F1 score of AdaBoost: ', f1)

def runKNN(n, weight): # weight --> string!
	# K-Nearest Neighbors
	knn = KNeighborsClassifier(n_neighbors=n, weights=weight, n_jobs=-1)
	print("=========== n_estimators = " + str(n) + " weight = " + weight + " ============")
	knn.fit(X_train, y_train)
	predicted = knn.predict(X_test)
	accuracy = knn.score(X_test, y_test)
	print('Accuracy of K-Nearest Neighbors: ', accuracy)
	f1 = f1_score(np.array(y_test), np.array(predicted), average="macro")
	print('F1 score of K-Nearest Neighbors: ', f1)
	# change k-value --> plot... how about weight?


# Stochastic Gradient Descendent.  def runSGD(loss_func, penalty_func, iteration):
# loss --> log, modified_huber, squared_hinge, perceptron, hinge(default)
# penalty --> 'l2(default), l1'
# alpha set as default 0.0001

# loss_list = ['log', 'modified_huber', 'squared_hinge', 'perceptron', 'hinge']
# p_list = ['l2', 'l1']
# #for l in loss_list:
for n in range(440, 520, 20):
	#for p in p_list:
		runSGD('hinge', 'l1', n)

# for n in range(500, 3500, 500):
# 	runRandomForest(n)

#for rate in np.arange(0.1, 0.2, 0.1):


# #runAdaBoost(1000, 0.05)
# num = [600, 800, 1000]
# for n in num:
# 	runAdaBoost(n, 0.05)

# W = ['uniform', 'distance']
# for w in W:
# 	for n in range(3, 11, 2):
# 		runKNN(n, w)


'''
# Multilayer-Perceptron
mlp = MLPClassifier(random_state=1, max_iter=300)
mlp.fit(X_train, y_train)
accuracy = mlp.score(X_test, y_test)
print('Accuracy of MLP: ', accuracy)
f1 = f1_score(np.array(y_test), np.array(predicted), average="macro")
print('F1 score of MLP: ', f1)





/Users/chpark/opt/anaconda3/lib/python3.8/site-packages/sklearn/linear_model/_stochastic_gradient.py:570: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
  warnings.warn("Maximum number of iteration reached before "
Accuracy of Stochastic Gradient Descendent:  0.5094715852442672
F1 score of Stochastic Gradient Descendent:  0.5195506451720924
================== max_iter = 180 ==================
Accuracy of Stochastic Gradient Descendent:  0.5358923230309073
F1 score of Stochastic Gradient Descendent:  0.5349723682952789
================== max_iter = 200 ==================
Accuracy of Stochastic Gradient Descendent:  0.5548354935194417
F1 score of Stochastic Gradient Descendent:  0.5523442405530626
================== max_iter = 220 ==================
Accuracy of Stochastic Gradient Descendent:  0.5398803589232303
F1 score of Stochastic Gradient Descendent:  0.5380874233131921
================== max_iter = 240 ==================
Accuracy of Stochastic Gradient Descendent:  0.39232303090727816
F1 score of Stochastic Gradient Descendent:  0.4264611508545489
================== max_iter = 260 ==================
Accuracy of Stochastic Gradient Descendent:  0.47557328015952144
F1 score of Stochastic Gradient Descendent:  0.4786684714157211
================== n_estimators = 500 ==================
Accuracy of Random Forest:  0.6674975074775673
F1 score of Random Forest:  0.66982529436123
================== n_estimators = 1000 ==================
Accuracy of Random Forest:  0.672482552342971
F1 score of Random Forest:  0.6763421449815104
================== n_estimators = 1500 ==================
Accuracy of Random Forest:  0.6679960119641076
F1 score of Random Forest:  0.6689828178140351
================== n_estimators = 2000 ==================
Accuracy of Random Forest:  0.6669990029910269
F1 score of Random Forest:  0.6681245930693382
================== n_estimators = 200 with learning_rate = 0.1 ==================
Accuracy of AdaBoost:  0.34496510468594216
F1 score of AdaBoost:  0.3278340822585036
================== n_estimators = 600 with learning_rate = 0.1 ==================
Accuracy of AdaBoost:  0.32452642073778665
F1 score of AdaBoost:  0.32059155787120325

'''
