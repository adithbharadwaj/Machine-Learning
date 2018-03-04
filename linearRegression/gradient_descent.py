
# simple implementation of linear regression for only one feature/variable. 

import numpy as np
import matplotlib.pyplot as plt 

from sklearn.linear_model import LinearRegression # using sklearn's built in regression 

def calculate_summation(theta1, theta2, points): # calculates the summation term (sigma) 1/n*(h(x) - y) 

	sum1 = 0
	sum2 = 0

	for i in range(len(points)):
		 # getting each point, which is a pair of (x, y)
		x = points[i, 0]
		y = points[i, 1]
		
		hypothesis = (theta1 + theta2*x)	
		
		sum1 += (hypothesis - y)
		sum2 += ((hypothesis - y) * x)


	return sum1/len(points), sum2/len(points)


def gradient_descent(theta1, theta2, points, alpha, numOfIterations):
		
	cost = list()
	theta_plot = list()

	for i in range(numOfIterations):
		
		temp0, temp1 = calculate_summation(theta1, theta2, np.array(points)) # temp1 and temp2 are the summation terms
		theta1 = theta1 - (alpha) * temp0
		theta2 = theta2 - (alpha) * temp1
		cost.append(cost_function(theta1, theta2, np.array(points)))
		theta_plot.append(theta1)

	return theta1, theta2, cost, theta_plot


def cost_function(theta1, theta2, points):  # function to calculate the cost function j.

	m = len(points)
	j = 0

	for i in range(len(points)):
		
		x = points[i, 0]
		y = points[i, 1]

		hypothesis = (theta1 + theta2*x)

		j += 1/(2*m) * (hypothesis - y)**2 #calculating the hypothesis function

	return j	


def predict(theta1, theta2, x):
	
	y = theta1 + theta2 * x # predicting the values by using the minimized value of theta1 and theta2.
	return y 	


if __name__ == '__main__':

	initial_theta1 = 0 
	initial_theta2 = 0

	numOfIterations = 10000

	points = np.genfromtxt("data.csv", delimiter=",") #using numpy to get a csv file having two columns corresponding to x and y resp.

	learning_rate = 0.0003

	theta_1 , theta_2, j, theta_plot = gradient_descent(initial_theta1, initial_theta2, points, learning_rate, numOfIterations)

	regression_line = [theta_2 * i + theta_1 for i in range(len(points))] # finding the regression line by using theta1 and theta2. 

	print("the value of c or theta0 is: ",theta_1, "the value of theta1 is " , theta_2, " minimum value of cost function ", j[len(j) - 1])

	# using matplotlib's pyplot

	plt.subplot(121)
	plt.scatter(points[:, 0], points[:, 1]) # plotting the data points as a scatter plot
	plt.plot(regression_line, 'r') # plotting the regression line over the scatter plot of data points
	plt.xlabel('x values of data')
	plt.ylabel('y values of data')

	# plotting the cost function against theta1

	plt.subplot(122)
	plt.plot(theta_plot, j)
	plt.xlabel('the values of theta 1')
	plt.ylabel('the values of cost function j')
	plt.legend(['cost function'])

	# the same linear regression model can be simplified by using sklearn

	scikitlearn_regression = LinearRegression()

	x = np.array(points[:, 0])
	x = x.reshape(-1, 1)
	y = np.array((points[:, 1]))

	x_train = x[:int(len(x)/2) + 25]
	y_train = y[:int(len(y)/2) + 25]

	x_test = x[int(len(x)/2) + 25:]
	y_test = y[int(len(y)/2) + 25:]

	scikitlearn_regression.fit(x_train, y_train) #splitting data into training and testing values

	print("theta1, theta2 are ",scikitlearn_regression.coef_, scikitlearn_regression.intercept_) # sklearn's intercept and slope values. 
	
	print("r squared score for test: ", scikitlearn_regression.score(x_test, y_test))
	print("r square score for train ", scikitlearn_regression.score(x_train, y_train))

	y_pred = scikitlearn_regression.predict(x[int(len(x)/2):])

	regression_line_2 = [scikitlearn_regression.coef_ * i + scikitlearn_regression.intercept_ for i in range(len(points))]

	plt.subplot(121)
	plt.plot(regression_line_2, 'g')
	plt.legend(['regression using my code', 'regression sklearn' ,'data', ])

	plt.show()
