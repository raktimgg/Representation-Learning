######################## Coded in python 2.7 #################

import numpy as np
import matplotlib.pyplot as plt 

# def sort(x):
# 	index = np.zeros((x.shape[0]),dtype = 'int32')
# 	for i in range(0,x.shape[0]-1):
# 		index[i] = i
# 		for j in range(i+1,x.shape[0]):
# 			if x[j]>x[index[i]]:
# 				temp = index[i]
# 				index[i] = j
# 				index[j] = i
# 			# print index[i]
# 		temp = x[i]
# 		x[i] = x[index[i]]
# 		x[index[i]] = temp
# 	# print x
# 	return x, index


x1 = plt.imread('image.jpeg')
# print x1.shape
x = x1.reshape(x1.shape[2],x1.shape[0]*x1.shape[1])
# print x.shape
x2 = np.empty(x.shape)
for i in range(0,x.shape[0]):
	m= np.mean(x[i,:])
	x2[i,:] = x[i,:] - m*np.full((x.shape[1]),1.)
# print x2.shape, np.mean(x2[0,:])
Cxx = (1./x2.shape[1])*np.dot(x2,x2.T)
eig, vec = np.linalg.eig(Cxx)
# print vec1
# print eig_sort
P = vec.T
Cyy = np.eye(eig.shape[0])*eig
print "Covariance of Y from eigen values of Cxx = "
print Cyy
print " "
y = np.dot(P,x2)
print "Y = "
print y
print " "
# print eig, vec
print "Covariance of Y using (1/n)(Y.Ytrans) =  "
Cyy1 =  np.dot(np.dot(P,Cxx),P.T)
print Cyy1
# print y1
# print y.shape

# y1 = y.reshape(x1.shape[0],x1.shape[1],x1.shape[2])    ###################### Uncomment to plot picture ########################
# print y1.shape
# plt.imshow(y1)
# plt.show()