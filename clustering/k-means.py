
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

data = loadmat('ex7data2.mat') # loading the image data.
A = data['X']

points = np.array(A)

clusters = 3 # no. of clusters.

means = np.zeros((clusters, 2)) # means or centroids.

for i in range(clusters):
	rand1 = int(np.random.random(1)*10)
	rand2 = int(np.random.random(1)*8)
	means[i, 0] = points[rand1, 0]
	means[i, 1] = points[rand2, 1]

def distance(x1, y1, x2, y2):
	dist = np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))
	return dist

flag = 10

index = np.zeros(A.shape[0])

#k-means algorithm.

while(flag > 0):
	for j in range(len(points)):
		minv = 1000
		temp = -1
		for k in range(clusters):
			x1 = points[j, 0]
			y1 = points[j, 1]
			x2 = means[k, 0]
			y2 = means[k, 1]
			if(distance(x1, y1, x2, y2) < minv):
				minv = distance(x1, y1, x2, y2)
				temp = k
				index[j] = k	
	
	for k in range(clusters):
		sumx = 0
		sumy = 0
		count = 0
		for j in range(len(points)):
			if(index[j] == k):
				sumx += points[j, 0]
				sumy += points[j, 1] 
				count += 1
		if(count == 0):
			count = 1		
		means[k, 0] = float(sumx/count)
		means[k, 1] = float(sumy/count)		
		
		'''
		plt.scatter(points[:, 0], points[:, 1])
		plt.scatter(means[:, 0], means[:, 1])
		plt.show()
		'''
	flag -= 1


cluster1 = points[np.where(index == 0)[0],:]  
cluster2 = points[np.where(index == 1)[0],:]  
cluster3 = points[np.where(index == 2)[0],:]

#plotting the clusters.

fig, ax = plt.subplots(figsize=(12,8))  
ax.scatter(cluster1[:,0], cluster1[:,1], s=30, color='r', label='Cluster 1')  
ax.scatter(cluster2[:,0], cluster2[:,1], s=30, color='g', label='Cluster 2')  
ax.scatter(cluster3[:,0], cluster3[:,1], s=30, color='b', label='Cluster 3')  
ax.legend() 
plt.show()
