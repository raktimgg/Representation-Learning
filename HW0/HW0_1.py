################# Coded in python 2.7 #####################

import numpy as np
import matplotlib.pyplot as plt

def findindex(x,s):
	small = np.empty(s.shape[1])
	for i in range(0,s.shape[1]):
		small[i] = np.linalg.norm(x-s[:,i])
	return np.argmin(small)


def plotclusters(x,k,nos):
	temp = np.empty([nos,3])
	M = 0
	for i in range(0,k):
		l = len(x[i])
		for j in range(0,l):
			# print x[i][j]
			temp[M,:] = x[i][j]
			M+=1
	return temp

x1 = plt.imread('image.jpg')
# print x1.shape
x = x1.reshape(x1.shape[2],x1.shape[1]*x1.shape[0])
# print x.shape
# nos = x.shape[1]
nos = 1000                    ######################## Only 1000 elements are taken to reduce time in computing ########################
x = x[:,:nos]
print "Enter the number of clusters"
k = input()
# k = 5

s = np.empty([x.shape[0],k])
n = x.shape[1]/k
j = 0

############# Initializing k as mean of every nth element ##################
for i in range(0,s.shape[1]):
	s[:,i] = np.mean(x[:,j:j+n])
	j = j+n
index = np.empty(x.shape[1])

j = 0
error = 100
epsilon = 0.001
while(error>epsilon):
	j+=1
	error = 0
	temp = np.empty([s.shape[0],s.shape[1]])
	temp[:,:] = s[:,:]
	cluster = [[] for i in range(k)]
	for i in range(0,x.shape[1]):
		index = findindex(x[:,i],s)
		cluster[index].append(x[:,i])
	cluster = np.array(cluster)
	for i in range(0,s.shape[1]):
		x1 = np.array(cluster[i])
		if(x1.shape[0]>0):
			s[:,i] = np.mean(x1,axis = 0)
		else:
			s[:,i] = np.random.rand(3)
	for i in range(0,s.shape[1]):
		error = error + np.linalg.norm(s[:,i]-temp[:,i])
	print "Epoch = ", j,"Error = " ,error

print "Centroids are "
print s.T


##################### Uncomment the following 3 lines and change index accordingly to print clusters ##############################

# index = 0
# print "The Clusters are "
# print cluster[index]

###################################################################################################################################


#################### Uncomment the following lines to plot the 2D representation of the image after clustering #########################
#################### The location of the points will be very far from one another                              #########################

# pl = plotclusters(cluster,k,nos)
# print pl.shape
# print pl[0]
# f = 10
# pl = pl.reshape(pl.shape[0]/f,f,pl.shape[1])
# plt.imshow(pl[:100]) ###################### I have printed only till 100 to make it visible without zoming in ###################
# plt.show()




###################################### UNcomment this to print centroids ##########################

# plt.imshow(s)
# plt.show()