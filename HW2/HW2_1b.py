import numpy as np


x_train = np.array([[1., 1.],[1., 0.],[0., 1.],[0., 0.]])
y_train = np.array([[1.],[0.],[0.],[0.]])
# print(x_train.shape)

def mse(y_hat, y):
	error = 0
	for i in range(0,y.shape[0]):
		error+=np.sum((y_hat[i]-y[i])**2)
	return error/i

def dense(x,w):
	return np.matmul(w,x)

def sigmoid(x):
	return (1/(1+np.exp(-1.0*x)))

def sigmoidprime(x):
	return sigmoid(x)*(1-sigmoid(x))



def train(x_tr, y_tr, hidden_nodes, output_nodes, noe, lr):
	w1 = np.random.rand(hidden_nodes,x_tr.shape[1])
	b1 = np.random.rand(hidden_nodes)
	w2 = np.random.rand(output_nodes,hidden_nodes)
	b2 = np.random.rand(output_nodes)
	print w1, b1, w2, b2
	y_hat = np.empty([y_tr.shape[0],1])
	for epochs in range(0,noe):
		for i in range(0,y_tr.shape[0]):
			for k in range(0,y_tr.shape[1]):
				hidden = dense(x_tr[i],w1) + b1
				# print hidden.shape
				hidden_act = sigmoid(hidden)
				out = dense(hidden_act, w2) + b2
				y_hat[i][k] = sigmoid(out)[0]
				# print y_hat.shape
			for a in range(0,w2.shape[0]):
				for b in range(0,w2.shape[1]):
					w2[a][b] = w2[a][b] - lr*2*(y_hat[i][a]-y_tr[i][a])*sigmoidprime(out[a])*hidden_act[b]
					b2[a] = b2[a] - lr*2*(y_hat[i][a]-y_tr[i][a])*sigmoidprime(out[a])

			for a in range(0,w1.shape[0]):
				for b in range(0, w1.shape[1]):
					w1[a][b] = w1[a][b] - 2*np.sum((y_hat[i]-y_tr[i])*sigmoidprime(out)*w2[:,a])*sigmoidprime(hidden[a])*x_tr[i][b]
					b1[a] = b1[a] - 2*np.sum((y_hat[i]-y_tr[i])*sigmoidprime(out)*w2[:,a])*sigmoidprime(hidden[a])

		error = mse(y_hat,y_tr)
		print error
	return(w1,w2,b1,b2)
			

noh = 2
nou = 1
noe = 10000
lr = 0.1

w1,w2,b1,b2 = train(x_train,y_train,noh,nou,noe,lr)  


############################################# Uncomment this part to test it #######################################################################

inp = np.empty(x_train.shape[1])
t = 1
while(t):
	print("Enter test values")
	for i in range(0,inp.shape[0]):
		inp[i] = input()
	out = (sigmoid(np.matmul(w2,sigmoid(np.matmul(w1,inp)+b1))+b2))
	if(out>=0.5):
		print("Output is 1")
	else:
		print("Output is 0")