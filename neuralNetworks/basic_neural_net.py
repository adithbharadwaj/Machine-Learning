
''' 
   implementing basic neural networks for performing basic boolean operations
   such as AND, OR, NOR, XNOR.

   AND: weights = [20, 20], bias = [-30]
   OR: weights = [20, 20], bias = [-10]
   NOR: weights = [-20, -20], bias = [10]

   XNOR is obtained by ORing the output of AND neuron with the output of NOR neuron.

'''

import numpy as np


def sigmoid(x):
	return 1/(1 + np.exp(-x))


def hypothesis(theta, X):

	X = np.array(X)
	row = X.shape[0]	     						# shape[0] gives the number of rows of X
	ones = np.ones((row, 1))						# creating a column vector of ones
	X = np.concatenate((ones, X), axis = 1)          # adding ones to the bias term since it is not multiplied by the x term. 
													# axis = 1 specifies column concatenation.			
	theta = np.array(theta)
	
	return np.dot(X, theta.T)  # hypothesis : bias*1 + x1*weight1 + x2*weight2 ...... 


def AND(X):
	
	weights = [20, 20]
	bias = [-30]
	theta = bias + weights # theta = [-30, 20, 20]

	hyp = hypothesis(theta, X)
	ans = sigmoid(hyp)

	return np.round(ans)

def OR(X):

	weights = [20, 20]
	bias = [-10]
	theta = bias + weights   # theta = [-10, 20, 20]

	hyp = hypothesis(theta, X)
	ans = sigmoid(hyp)

	return np.round(ans)

def NOR(X):

	weights = [-20, -20] 
	bias = [10]
	theta = bias + weights  # theta = [10, -20, -20]

	hyp = hypothesis(theta, X)
	ans = sigmoid(hyp)

	return np.round(ans)

def XNOR(X):                # here, the output of and and nor neuron is ORed to get the output
	
	nor_output = NOR(X)
	and_output = AND(X)

	# concatenating the output of NOR and AND by reshaping it into a column vector. 

	new_X = np.concatenate((nor_output.reshape(-1, 1), and_output.reshape(-1, 1)), axis = 1) # reshaping it into column vector
	
	ans = OR(new_X) 			# computing the or of the new value of X so that the final ans is the XNOR value
 
	return ans


if __name__ == '__main__':

	X = [[0, 0],
 	 	 [0, 1], 
 	 	 [1, 0], 
 	 	 [1, 1]]

	print("AND is", AND(X))
	print("OR is", OR(X))
	print("NOR is",NOR(X))
	print("XNOR is", XNOR(X))



