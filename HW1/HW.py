############################################################################
#                                                                          
#						Run the code in python 2
#
############################################################################
import numpy as np


def gmm_sum(x,mu,sig):					################ For findinf sum in the posterior term ##############
	res = np.zeros(mu.shape[0])
	for i in range(0,mu.shape[0]):
		res[i] = gmm(x,mu[i],sig[i])
	return res


def vector_mul(a,b):					################## Function for vector multiplication ##############
	res = np.zeros([a.shape[0],b.shape[0]])
	for i in range(0,a.shape[0]):
		for j in range(0,b.shape[0]):
			res[i][j] = a[i]*b[j]
	return res



def gmm(x,mu,sig):						################## Function for finding PDF of multivariate nonrmal #######
	d = x.shape[0]
	k = mu.shape[0]
	const = 1/np.sqrt(((2*np.pi)**d)*np.linalg.det(sig))
	res = const*np.exp((-1./2)*np.dot((x-mu).T,np.dot(np.linalg.inv(sig),(x-mu))),dtype = 'float64')
	return res


d = 3
N = 100
x = np.random.normal(100,10,(d,N)) ########## Defining a dxN variable #################################

print ("Enter the value of K")
K = input()
# K = 5

pi = np.ones(K)*1./K
mu = np.zeros([K,d])
j = 0
hop = int(N/K)

#################### Initializing mu #############################


for i in range(0,mu.shape[0]):
	mu[i,:] = np.mean(x[:,j:j+hop])
	j = j+hop

sig = np.zeros([K,d,d])
j = 0
for i in range(0,sig.shape[0]):
	sig[i,:,:] = np.cov(x[:,j:j+hop])
	j = j+hop

################### INitializing covariance ##########################

gama = np.zeros([K,N])
N_temp = np.zeros(K)

error = 100
epsilon = 1e-4
log_error = 0
a = 0



while(error>epsilon):
	for k in range(0,K):
		for n in range(0,N):
			temp = gmm_sum(x[:,n],mu,sig)
			post_sum = np.dot(pi.T,temp)
			gama[k,n] = (pi[k]*gmm(x[:,n],mu[k],sig[k]))*1.0/post_sum
		
		################### Updating Nk ########################

		N_temp[k] = np.sum(gama[k],axis = 0)

		################### Updating mean ######################
		total = np.zeros(d)
		for n in range(0,N):
			total = total + gama[k,n]*x[:,n]
		mu[k,:] = total/N_temp[k]

		################### Updating sigma #####################
		total = np.zeros([d,d])
		for n in range(0,N):
			total[:,:] = total[:,:] + gama[k,n]*vector_mul((x[:,n]-mu[k,:]).T,(x[:,n]-mu[k,:]))
		sig[k,:,:] = total/N_temp[k]
		pi[k] = N_temp[k]*1.0/N
	total = 0
	for n in range(0,N):
		post_sum = np.dot(pi.T,gmm_sum(x[:,n],mu,sig))
		total = total + np.log(post_sum)

	########################### Finding error #######################

	error = np.abs(total - log_error)
	print ("Iteration =", a,"Error = ", error)
	a+=1
	log_error = total

print (" The value of mu = ") 
print mu

print (" The value of covariance = ") 
print sig