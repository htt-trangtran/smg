############################
# written by Lam M. Nguyen and Trang H. Tran
############################

"""
Useful functions (Logistic regression + Nonconvex)
"""

import numpy as np
import random

# Sigmoid function
def sigmoid(z):  
    return 1 / (1 + np.exp(-z))

# Computing accuracy
def accuracy(n, d, X, Y, w, lamb):
	acc = 0
	for i in range(0,n): 
		w_Xi = np.dot(w,X[i,:])
		if (Y[i]*w_Xi >= 0):
			acc = acc + 1
	acc_val = acc/(n+0.0)
	return acc_val

# # Computing function value
# def func_val(n, d, X, Y, w, lamb): 
# 	loss = 0
# 	for i in range(0,n): 
# 		w_Xi = np.dot(w,X[i,:])
# 		exp_YwX = np.exp(-Y[i]*w_Xi)
# 		if (exp_YwX > 1e20): 
# 			loss = loss + 20 / (n + 0.0) 
# 		else: 
# 			loss = loss + np.log(1 + exp_YwX) / (n + 0.0) 
# 	loss = loss + (lamb/(2+0.0))*np.dot(w,w)		
# 	return loss

# Computing function value
def func_val(n, d, X, Y, w, lamb): 
	loss = 0
	for i in range(0,n): 
		w_Xi = np.dot(w,X[i,:])
		exp_YwX = np.exp(-Y[i]*w_Xi)
		if (exp_YwX > 1e20): 
			loss = loss + 20 / (n + 0.0) 
		else: 
			loss = loss + np.log(1 + exp_YwX) / (n + 0.0) 

	w_sq = np.square(w)
	reg = np.sum(np.divide(w_sq, 1 + w_sq))

	loss = loss + (lamb/(2+0.0)) * reg		
	return loss
	
# # Computing component gradient value
# def grad_com_val(i, n, d, X, Y, w, lamb):
# 	w_Xi = np.dot(w,X[i,:])
# 	exp_YwX = np.exp(-Y[i]*w_Xi)
# 	if (exp_YwX > 1e20):
# 		grad = -Y[i] * X[i,:] + lamb * w
# 	else: 
# 		grad = (-Y[i] * X[i,:] * exp_YwX / ( 1 + exp_YwX)) + lamb * w
	
# 	return grad

# Computing component gradient value
def grad_com_val(i, n, d, X, Y, w, lamb):
	w_Xi = np.dot(w,X[i,:])
	exp_YwX = np.exp(-Y[i]*w_Xi)
	if (exp_YwX > 1e20):
		grad = -Y[i] * X[i,:]
	else: 
		grad = (-Y[i] * X[i,:] * exp_YwX / ( 1 + exp_YwX))
	
	w_sq = np.square(w)
	grad_reg = np.divide(w, np.square(1 + w_sq))

	grad = grad + lamb * grad_reg

	return grad
	
# Computing full gradient
def full_grad_eval(n, d, X, Y, w, lamb):
	full_grad = np.zeros(d)
	for i in range(0,n):
		full_grad = full_grad + grad_com_val(i, n, d, X, Y, w, lamb)
	
	full_grad = full_grad/(n+0.0)
	
	return full_grad



# Computing batch gradient (without replacement)
def batch_grad_eval(n, d, X, Y, w, lamb, batchsize):
	batch_grad = np.zeros(d)
	index = [i for i in range(0,n)]
	random.shuffle(index)

	for i in range(0,batchsize):
		batch_grad = batch_grad + grad_com_val(index[i], n, d, X, Y, w, lamb)

	batch_grad = batch_grad/(batchsize+0.0)

	return batch_grad

# Computing batch gradient (indices)
def batch_grad_indices_eval(indices, n, d, X, Y, w, lamb):
	batch_grad = np.zeros(d)
	batchsize = len(indices)

	for i in indices:
		batch_grad = batch_grad + grad_com_val(i, n, d, X, Y, w, lamb)

	batch_grad = batch_grad/(batchsize+0.0)

	return batch_grad


