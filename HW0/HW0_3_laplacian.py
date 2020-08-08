#################### Coded in python 2.7 #########################

import numpy as np
import matplotlib.pyplot as plt 

####################### Function to sort samples to find median ######################
def sort(x):
	index = np.zeros((x.shape[0]),dtype = 'int32')
	for i in range(0,x.shape[0]-1):
		index[i] = i
		for j in range(i+1,x.shape[0]):
			if x[j]>x[index[i]]:
				temp = index[i]
				index[i] = j
				index[j] = i
			# print index[i]
		temp = x[i]
		x[i] = x[index[i]]
		x[index[i]] = temp
	# print x
	return x, index   

############################# Function to find median ##############################
def median(a):
	x, s = sort(a)
	n = x.shape[0]
	if n%2 == 0:
		return (x[n/2]+x[(n/2)+1])*1.0/2
	else:
		return x[(n/2)]	  

def getparam(a):
	N = a.shape[0]
	mu1 = median(a)
	temp = (a - mu1*np.full((a.shape[0]),1.))
	lam1 = np.linalg.norm(a,1)*1.0/N
	return mu1, lam1

mu = 0
lam = 1.0
size = 1000
x = np.random.laplace(mu,lam,size)
mu_pred, lam_pred = getparam(x)
print "Given Value of mean = ", mu
print "Using MLE, value of mean = ", mu_pred
print "Given Value of Lambda = ", lam
print "Using MLE, value of Lambda = ", lam_pred

xn = np.random.laplace(mu,lam,size)


########################### Uncomment to plot histogram ######################################



# plt.hist(x,100,label ='Original Sample')
# plt.hist(xn,100,label ='Sample from predicted parameter')
# plt.legend()
# plt.show()