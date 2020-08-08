##################  Please run it on python 2 #############################

import numpy as np

from skimage.transform import resize
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST data/", one_hot=True)
x_train = mnist.train.images
del mnist
x_train = x_train[:10] ######################################### Only 10 samples have been taken to reduce running time. Due to this, the loss will fluctuate a bit during training ##################################
x_train = x_train.reshape(x_train.shape[0], 28,28)

x_train = resize(x_train, (x_train.shape[0], x_train.shape[1]//2, x_train.shape[2]//2), anti_aliasing = True) 
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
print (x_train.shape)

y_train = np.empty(x_train.shape)
y_train[:] = x_train[:]
def mse(y_hat, y):
	error = 0
	for i in range(0,y.shape[0]):
		error+=np.sum((y_hat[i]-y[i])**2)
	return error/i

def dense(x,w):
	return np.matmul(w,x)

def sigmoid(x):
	x = x.astype("float128")
	return (1/(1+np.exp(-1.0*x)))

def sigmoidprime(x):
	return sigmoid(x)*(1-sigmoid(x))



def train(x_tr, y_tr, hidden_nodes, output_nodes, noe, lr):
	p = 0.005
	lam = 0.1
	w1 = np.random.rand(hidden_nodes,x_tr.shape[1])
	b1 = np.random.rand(hidden_nodes)
	w2 = np.random.rand(output_nodes,hidden_nodes)
	b2 = np.random.rand(output_nodes)
	y_hat = np.empty([y_tr.shape[0],y_tr.shape[1]])
	hidden_act = np.empty([x_tr.shape[0], hidden_nodes])
	for epochs in range(0,noe):
		for i in range(0,y_tr.shape[0]):
			for k in range(0,y_tr.shape[1]):
				# print x_tr[i].shape, w1.shape, b1.shape, dense(x_tr[i],w1).shape
				hidden = dense(x_tr[i],w1) + b1
				# print hidden.shape
				hidden_act[i] = sigmoid(hidden)
				out = dense(hidden_act[i], w2) + b2
				# print out.shape
				y_hat[i][k] = sigmoid(out[0])
				# print y_hat.shape
		for a in range(0,w2.shape[0]):
			for b in range(0,w2.shape[1]):
				w2[a][b] = w2[a][b] - lr*np.sum(2*(y_hat[:,a]-y_tr[:,a])*sigmoidprime(y_hat[:,a])*hidden_act[i][b])
				b2[a] = b2[a] - lr*np.sum(2*(y_hat[:,a]-y_tr[:,a])*sigmoidprime(y_hat[:,a]))

		for a in range(0,w1.shape[0]):
			for b in range(0, w1.shape[1]):
				temp = (lam*(-(1*p/np.sum(hidden_act,axis = 1))+ (1-p)/(1-np.sum(hidden_act,axis =1))))*((1/y_tr.shape[0])*np.sum(sigmoidprime(hidden_act[:,a])*x_tr[:,b]))
				err1 = (np.sum((y_hat-y_tr)*sigmoidprime(y_hat)*w2[:,a],axis = 1)*sigmoidprime(hidden_act[i][a])*x_tr[:,b])
				w1[a][b] = w1[a][b] - (2*lr*np.sum(err1+ temp ))
				b1[a] = b1[a] - (2*lr*np.sum(np.sum((y_hat-y_tr)*sigmoidprime(y_hat)*w2[:,a],axis = 1)*sigmoidprime(hidden_act[i][a])))
			# print i
		error = mse(y_hat,y_tr)
		print ("error =", error, "Epoch =",  epochs+1)
	return(w1,w2,b1,b2)
			

hid_size = 50
out_size = y_train.shape[1]
epoch = 40 ########################################### Only 40 epochs have been taken to reduce running time ##############################
lr = 0.01

w1,w2,b1,b2 = train(x_train,y_train,hid_size,out_size,epoch,lr)


############################## Uncomment this to print the encoded image #################################################
#print w1, w2, b1, b2