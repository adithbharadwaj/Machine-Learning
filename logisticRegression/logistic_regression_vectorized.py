#logistic regression using vectorization

import math
import numpy as np
from sklearn.linear_model import LogisticRegression
import sklearn.datasets
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

def sigmoid(x):
	return 1/(1 + np.exp(-x))


def hypothesis(theta, x):
	z = np.dot(x, theta)
	return sigmoid(z)


def loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean() 


def gradient_descent(X, h, y):
	
	gradient = np.dot(X.T, (h - y))/len(y)
	return gradient

def logistic_regression(X, y, theta, alpha, numiters):	

	for i in range(numiters):

		h = hypothesis(theta, X)

		gradient = gradient_descent(X, h, y)

		theta = theta - (alpha) * gradient	
		
		h = hypothesis(theta, X)
		cost = loss(h, y)

	return theta, cost	


if __name__ == '__main__':

	data = np.loadtxt('ex2data1.txt', delimiter = ',')

	X = data[:, [0, 1]]
	y = data[:, 2]

	#iris = sklearn.datasets.load_iris()
	#X = iris.data[:, :2]
	#y = (iris.target != 0) * 1

	#plotting the given data

	pos = np.where(y == 1)
	neg = np.where(y == 0)

	plt.scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
	plt.scatter(X[neg, 0], X[neg, 1], marker = 'x', c ='r')
	plt.show()

	xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.5, random_state = 0)

	clf = LogisticRegression()
	clf.fit(xtrain, ytrain)

	res = clf.predict(xtest)

	score = 0

	#caclulating the score

	for i in range(len(res)):
		if(res[i] == ytest[i]):
			score += 1

	print("accuracy using manual calculation ", score/len(xtest))	

	from sklearn.metrics import accuracy_score # accuracy using sklearn

	print( "using sklearn's metric's accuracy_score function ",accuracy_score(res, ytest))	

	ones = np.ones(X.shape[0]).reshape(-1, 1)
	X = np.concatenate((ones, X), axis = 1)

	theta = np.zeros(X.shape[1])

	numiters = 15000

	alpha = 0.0001

	new_theta, cost = logistic_regression(X, y, theta, alpha, numiters)
	print(new_theta, cost)
	print(clf.coef_, clf.intercept_)