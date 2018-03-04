
# using sklearn's built in naive bayes implementation to classify data on vertebral problems as normal or abnormal

'''

data taken from the uci machine learning repository.

name of the file = 'vertebral.txt': 

col 1: pelvic_incidence numeric
col 2: pelvic_tilt numeric
col 3: lumbar_lordosis_angle numeric
col 4: sacral_slope numeric
col 5: pelvic_radius numeric
col 6: degree_spondylolisthesis numeric

labels : {Normal, Abnormal}

'''
import matplotlib.pyplot as plt # plot the data
import numpy as np 
from sklearn.naive_bayes import GaussianNB # sklearn's implementation of naive bayes.
import pandas as pd 
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('vertebral.txt') #using pandas to read the data since the features are floats and labels are columns. cant use numpy here
data = np.array(data) # converting the data to a numpy array to splice it in the next step(not possible without this line) 

x = data[:, :5] # features 
y = data[:, 6] # labels

ab = np.where(y == 'Abnormal') # finding out where the result is abnormal
normal = np.where(y == 'Normal') # finding out where the result is normal

# we can only plot two features using pyplot. so, here, i've chosen col 2 and 4. 
# any combination of columns can be used. 

plt.scatter(x[normal, 2], x[normal, 4], marker='o', c='r') # plotting the normal data
plt.scatter(x[ab, 2], x[ab, 4], marker = 'x', c = 'b') # plotting the abnormal data
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.75,random_state = 0) # splitting the data 

clf = GaussianNB()

clf.fit(x_train, y_train)
pred = clf.predict(x_test)

accuracy = accuracy_score(pred, y_test)

print(accuracy)

