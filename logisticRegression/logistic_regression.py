
# logistic regression implementation without using vectorization

import math
import numpy as np
from sklearn.linear_model import LogisticRegression
import sklearn.datasets
import matplotlib.pyplot as plt


def sigmoid(x):
	return 1/(1 + np.exp(-x))


def hypothesis(theta, x):
	
	h = 0
	for i in range(len(theta)):
		h += x[i] * theta[i]

	return sigmoid(h)		

def cost_function(theta, X, y):
	
	m = len(y)
	cost = 0

	for i in range(m):
		xi = X[i]       # ith row of the features matrix.
		h = hypothesis(theta, xi)

		cost += int(y[i]) * np.log(h) + (1 - int(y[i])) * np.log(1 - h)  

	constant = -(1/m)
	cost *= constant

	return cost 

def cost_function_derivative(theta, X, y, j):
	
	s = 0
	for i in range(len(y)):
		xi = X[i]
		hyp = hypothesis(theta, X[i])
		xij = xi[j]
		error = (hyp - int(y[i]))*xij
		s += error

	return s/len(y)	

def gradient_descent(theta, X, y, alpha):

	m = len(y)

	for j in range(len(theta)):
		derivative = cost_function_derivative(theta, X, y, j)
		theta[j] = theta[j] - (alpha) * derivative

	return theta	
	

def Logistic_regression(theta, X, y, alpha, numIter):


	for i in range(numIter):

		temp = gradient_descent(theta, X, y, alpha)
		theta = temp[:]
		cost = cost_function(theta, X, y)

	return theta, cost	

if __name__ == '__main__':

	data = np.loadtxt('ex2data1.txt', delimiter = ',')

	X = data[:, [0, 1]]
	y = data[:, 2]

	theta = [0, 0, 0]

	clf = LogisticRegression()
	clf.fit(X, y)

	z = np.ones(X.shape[0])
	z = z.reshape(-1, 1)
	X = np.concatenate((z, X), axis = 1)

	alpha = 0.0001
	numIter = 15000

	new_theta, cost = Logistic_regression(theta, np.array(X), y, alpha, numIter)
	print(new_theta, cost)












