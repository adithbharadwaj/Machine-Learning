
''' 
	3 layer neural network (can be used to compute XOR ) 
	
	*input layer has 2 input neurons (plus one for bias)
	*hidden layer has 2 neurons (plus one for bias)
	*output layer has one neuron

	note: number of layers can be changed depending on the inputs. 

'''

import numpy as np

def sigmoid(x):
	return (1/(1 + np.exp(-x)))

# derivative of the sigmoid function.

def sigmoid_derivative(x):
	return sigmoid(x)*(1 - sigmoid(x))

# function for forward propagation

def forward_propagation(inputs, weights1, weights2):

	ones = np.ones(inputs.shape[0])
	inputs = np.c_[ones, inputs]

	activation1 = sigmoid(np.dot(inputs, weights1)) # output of hidden layer 
	output = sigmoid(np.dot(activation1, weights2)) # output of output layer (final output) 

	return activation1, output

#function for back propagation. 	

def back_propagation(inputs, actual_outputs, network_outputs, weights1, weights2, a2):

	learning_rate = 2

	#error for output layer is calculated by (y - output) * sigmoid_derivative(output)
	#error for hidden layer is (d3 * weights2) * sigmoid_derivative(a2), where a2 is the output of hidden layer

	d3 = np.multiply(sigmoid_derivative(network_outputs), (actual_outputs - network_outputs))
	d2 = np.multiply(sigmoid_derivative(a2), np.dot(d3, weights2.T))

	#delta error is learning_rate * (inputs_for_respective_layer * error_for_respective_layer)

	del3 = np.multiply(learning_rate, np.dot(a2.T, d3))
	del2 = np.multiply(learning_rate, np.dot(inputs.T, d2))

	# adjusting the weights by adding them with the delta value. 

	weights1 += del2
	weights2 += del3

	return weights1, weights2


def train(inputs, weights1, weights2, num_iter, actual_output):
	
	for i in range(num_iter):

		#forward prop

		a2 = sigmoid(np.dot(inputs, weights1))
		output = sigmoid(np.dot(a2, weights2))

		learning_rate = 2

		# back prop

		d3 = np.multiply(sigmoid_derivative(output), (actual_output - output))
		d2 = np.multiply(sigmoid_derivative(a2), np.dot(d3, weights2.T))

		del3 = np.multiply(learning_rate, np.dot(a2.T, d3))
		del2 = np.multiply(learning_rate, np.dot(inputs.T, d2))

		weights1 += del2
		weights2 += del3

	return weights1, weights2	


if __name__ == '__main__':

	np.random.seed(1) # giving the seed so that the random values are the same everytime. 

	training_set_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
	training_set_outputs = np.array([[0, 1, 1, 0]]).T

	#adding ones to the input for the bias unit. 

	ones = np.ones(training_set_inputs.shape[0])
	inputs = np.c_[ones, training_set_inputs]

	epsilon = 1

	# number of layers. 

	input_layers = 2
	hidden_layer = 2
	output_layer = 1

	# random initialization of weights. plus 1 for bias unit. 
	# weights lie in the range [-epsilon, epsilon]

	weights1 = 2*epsilon*np.random.rand(input_layers + 1, hidden_layer + 1) - epsilon
	weights2 = 2*epsilon*np.random.rand(hidden_layer + 1, output_layer) - epsilon

	ans_weight1, ans_weight2 = train(inputs, weights1, weights2, 10000, training_set_outputs) #training the net

	dum, pred = forward_propagation(np.array([[0, 1]]), ans_weight1, ans_weight2) #predicting the output 

	print(np.round(pred))