#################### Coded in python 2.7 #########################

import numpy as np
import matplotlib.pyplot as plt 

def getparam(a):
	N = a.shape[0]
	return np.sum(a)*1.0/N

lam = 1.0
size = 100
x = np.random.poisson(lam,size)
param = getparam(x)
print "Value of lambda given = ", lam
print "Value of lambda by using MLE = ", param
xn = np.random.poisson(param,size)

########################### Uncomment to plot histogram ######################################


# plt.hist(x,100,label ='Original Sample')
# plt.hist(xn,100,label ='Sample from predicted parameter')
# plt.legend()
# plt.show()