#################### Coded in python 2.7 #########################

import numpy as np 
import matplotlib.pyplot as plt 

def getparameter(a,n):
	N = a.shape[0]
	return np.sum(a)*1.0/(n*N)

n = 1000
p = 0.5
size = 10000
x = np.random.binomial(n,p,size)
# print x.shape
param = getparameter(x,n)
print "From our definition of binomial, p = ", p
print "From MLE, estimated value of p = ", param

xn = np.random.binomial(n,param,size)

########################### Uncomment to plot histogram ######################################

# plt.hist(x,100,label ='Original Sample')
# plt.hist(xn,100,label ='Sample from predicted parameter')
# plt.legend()
# plt.show()
