#################### Coded in python 2.7 #########################

import numpy as np
import matplotlib.pyplot as plt 

def getparam(a):
	N = a.shape[0]
	return (N*1.0)/np.sum(a)

lam = 2.0
scale = 1/lam
size = 10000
x = np.random.exponential(scale,size)    ################# The functions acceps scale as a parameter and not lambda ####################
param = getparam(x)
print "Value of lambda given = ", lam
print "Value of lambda by using MLE = ", param
s1 = 1/param
xn = np.random.exponential(s1,size)


########################### Uncomment to plot histogram ######################################



# plt.hist(x,1000,label ='Original Sample')
# plt.hist(xn,1000,label ='Sample from predicted parameter')
# plt.legend()
# plt.show()