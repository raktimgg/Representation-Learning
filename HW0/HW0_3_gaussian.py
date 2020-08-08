#################### Coded in python 2.7 #########################

import numpy as np
import matplotlib.pyplot as plt 

def getparam(a):
	N = a.shape[0]
	mu1 = np.sum(a)*1.0/N
	temp = (a - mu1*np.full((a.shape[0]),1.))*(a - mu1*np.full((a.shape[0]),1.))
	sig1 = np.sum(temp)*1.0/N
	return mu1, sig1

mu = 0
sig = 2
size = 10000
x = np.random.normal(mu,sig,size)
mu_pred, sig_pred = getparam(x)
print "Given Value of mean = ", mu
print "Using MLE, value of mean = ", mu_pred
print "Given Value of standard deviation = ", sig*sig
print "Using MLE, value of standard deviation = ", sig_pred

xn = np.random.normal(mu,np.sqrt(sig_pred),size)

########################### Uncomment to plot histogram ######################################



# plt.hist(x,1000,label ='Original Sample')
# plt.hist(xn,1000,label ='Sample from predicted parameter')
# plt.legend()
# plt.show()