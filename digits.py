from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
import scipy
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random

import pickle

import os
from scipy.io import loadmat

#Load the MNIST digit data
M = loadmat("mnist_all.mat")


def softmax(y):
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    return exp(y)/tile(sum(exp(y),0), (len(y),1))


def tanh_layer(y, W, b):    
    '''Return the output of a tanh layer for the input matrix y. y
    is an NxM matrix where N is the number of inputs for a single case, and M
    is the number of cases'''
    return tanh(dot(W.T, y)+b)


def forward(x, W0, b0, W1, b1):
    # runs an input x through a single hidden-layer nn
    L0 = tanh_layer(x, W0, b0)
    L1 = dot(W1.T, L0) + b1
    output = softmax(L1)
    return L0, L1, output


def cross_entropy(y, y_):
    # cross entropy cost function
    return -sum(y_*log(y)) 


def deriv_multilayer(W0, b0, W1, b1, x, L0, L1, y, y_):
    # used to compute gradient using backpropagation
    dCdL1 = y - y_
    dCdW1 = dot(L0, dCdL1.T )
    

def part1():
    np.random.seed(0)
    for seed in range (0, 10):
        fig = figure()
        # array up to 5421
        for i in range(0, 10):
            index = np.random.randint(0, 5421)
            ax = fig.add_subplot(1+(i//5),5, 1+(i%5))
            imshow(M["train"+str(seed)][index].reshape((28,28)), cmap=cm.gray)
            plt.axis('off')
        plt.savefig("report/10_img_of_"+str(seed)+".jpg", bbox_inches='tight')
    # show()


def calculate_output(w, x):
    x = vstack( (ones((1, x.shape[1])), x))
    out = dot(w.T, x)
    return softmax(out)


def part2():
    # wx + b
    np.random.seed(0)
    w = 2*np.random.random((785, 10))-1
    x = np.random.random((1, 784))
    print(calculate_output(w, x))


def f(x, y_, w):
    # cost function y_ * log(softmax(y))
    p = calculate_output(w, x)
    return cross_entropy(p, y_)


def df(x, y_, w):    
    p = calculate_output(w, x)
    x = vstack( (ones((1, x.shape[1])), x))
    return dot(x,(p-y_).T)


def accuracy(x, y_, w):
    p = calculate_output(w, x)
    prediction = np.argmax(p, axis=0)
    target = np.argmax(y_, axis=0)
    return np.count_nonzero(prediction == target) / prediction.shape[0]


def grad_descent(f, df, x, y, init_t, alpha, learning_curve, xT, yT):
    EPS = 1e-5   # EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = 3000
    iter = 0
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha*df(x, y, t)         
        if iter % 25 == 0:
            print ("Iter", iter)
            cost = f(x, y, t)   
            acc_train = accuracy(x, y, t)
            acc_test = accuracy(xT, yT, t)
            learning_curve.append([iter, cost, acc_train, acc_test])
            print ("f(x) = ", cost) 
            print ("Accuracy train: ", acc_train)
            print ("Accuracy test: ", acc_test)
        iter += 1
    return t


def part3():
    n = 500
    x = M["train0"][0:n]
    x = vstack((x, M["train1"][0:n]))
    x = vstack((x, M["train2"][0:n]))
    x = vstack((x, M["train3"][0:n]))
    x = vstack((x, M["train4"][0:n]))
    x = vstack((x, M["train5"][0:n]))
    x = vstack((x, M["train6"][0:n]))
    x = vstack((x, M["train7"][0:n]))
    x = vstack((x, M["train8"][0:n]))
    x = vstack((x, M["train9"][0:n]))
    x = x.T/255.0
    
    y_ = []    
    for i in range(0, n):
        y_.append([1,0,0,0,0,0,0,0,0,0])        
    for i in range(0, n):
        y_.append([0,1,0,0,0,0,0,0,0,0])        
    for i in range(0, n):
        y_.append([0,0,1,0,0,0,0,0,0,0])        
    for i in range(0, n):
        y_.append([0,0,0,1,0,0,0,0,0,0])        
    for i in range(0, n):
        y_.append([0,0,0,0,1,0,0,0,0,0])        
    for i in range(0, n):
        y_.append([0,0,0,0,0,1,0,0,0,0])        
    for i in range(0, n):
        y_.append([0,0,0,0,0,0,1,0,0,0])        
    for i in range(0, n):
        y_.append([0,0,0,0,0,0,0,1,0,0])        
    for i in range(0, n):
        y_.append([0,0,0,0,0,0,0,0,1,0])         
    for i in range(0, n):
        y_.append([0,0,0,0,0,0,0,0,0,1])
        
    y_ = array(y_).T

    np.random.seed(0)
    w_init = np.random.random((785, 10))
    w = df(x, y_, w_init)
    
    delta = 0.0001
    dw = zeros((785, 10))
    dw[350,2] = delta
    
    print("Finite Difference: ")
    print((f(x,y_,w_init + dw) - f(x,y_,w_init-dw))/(2*delta))
    print("Direct derivative calculation")
    print(df(x,y_,w_init)[350,2])

    dw[150,7] = delta
    
    print("Finite Difference: ")
    print((f(x,y_,w_init + dw) - f(x,y_,w_init-dw))/(2*delta))
    print("Direct derivative calculation")
    print(df(x,y_,w_init)[150,7])
    

## PART 4
def part4():
    # use a smaller training set because training time is very long
    random.seed(1234)
    n = 500
    x = M["train0"][0:n]
    x = vstack((x, M["train1"][0:n]))
    x = vstack((x, M["train2"][0:n]))
    x = vstack((x, M["train3"][0:n]))
    x = vstack((x, M["train4"][0:n]))
    x = vstack((x, M["train5"][0:n]))
    x = vstack((x, M["train6"][0:n]))
    x = vstack((x, M["train7"][0:n]))
    x = vstack((x, M["train8"][0:n]))
    x = vstack((x, M["train9"][0:n]))
    x = x.T/255.0

    nT = n//10
    xT = M["test0"][0:nT]
    xT = vstack((xT, M["test1"][0:nT]))
    xT = vstack((xT, M["test2"][0:nT]))
    xT = vstack((xT, M["test3"][0:nT]))
    xT = vstack((xT, M["test4"][0:nT]))
    xT = vstack((xT, M["test5"][0:nT]))
    xT = vstack((xT, M["test6"][0:nT]))
    xT = vstack((xT, M["test7"][0:nT]))
    xT = vstack((xT, M["test8"][0:nT]))
    xT = vstack((xT, M["test9"][0:nT]))
    xT = xT.T/255.0
    
    # onehot encoding 
    y_ = []    
    for i in range(0, n):
        y_.append([1,0,0,0,0,0,0,0,0,0])        
    for i in range(0, n):
        y_.append([0,1,0,0,0,0,0,0,0,0])        
    for i in range(0, n):
        y_.append([0,0,1,0,0,0,0,0,0,0])        
    for i in range(0, n):
        y_.append([0,0,0,1,0,0,0,0,0,0])        
    for i in range(0, n):
        y_.append([0,0,0,0,1,0,0,0,0,0])        
    for i in range(0, n):
        y_.append([0,0,0,0,0,1,0,0,0,0])        
    for i in range(0, n):
        y_.append([0,0,0,0,0,0,1,0,0,0])        
    for i in range(0, n):
        y_.append([0,0,0,0,0,0,0,1,0,0])        
    for i in range(0, n):
        y_.append([0,0,0,0,0,0,0,0,1,0])         
    for i in range(0, n):
        y_.append([0,0,0,0,0,0,0,0,0,1])
        
    y_ = array(y_).T
    
    yT = []    
    for i in range(0, nT):
        yT.append([1,0,0,0,0,0,0,0,0,0])        
    for i in range(0, nT):
        yT.append([0,1,0,0,0,0,0,0,0,0])        
    for i in range(0, nT):
        yT.append([0,0,1,0,0,0,0,0,0,0])        
    for i in range(0, nT):
        yT.append([0,0,0,1,0,0,0,0,0,0])        
    for i in range(0, nT):
        yT.append([0,0,0,0,1,0,0,0,0,0])        
    for i in range(0, nT):
        yT.append([0,0,0,0,0,1,0,0,0,0])        
    for i in range(0, nT):
        yT.append([0,0,0,0,0,0,1,0,0,0])        
    for i in range(0, nT):
        yT.append([0,0,0,0,0,0,0,1,0,0])        
    for i in range(0, nT):
        yT.append([0,0,0,0,0,0,0,0,1,0])         
    for i in range(0, nT):
        yT.append([0,0,0,0,0,0,0,0,0,1])
        
    yT = array(yT).T

    # initial weights, bias is w[0]
    w_init = random.normal(0, 0.001, (785, 10))
    
    accuracies = []
    # learning step
    delta = 0.00002
    w = grad_descent(f, df, x, y_, w_init, delta, accuracies, xT, yT)
    accuracies = array(accuracies).T

    b = w[0:1].T
    w = w[1:].T

    # plot the learning curves
    
    plt.figure()
    plt.plot(accuracies[0], accuracies[2] * 100, 'g-', label = 'training set performance')
    plt.plot(accuracies[0], accuracies[3] * 100, 'r-', label = 'test set performance')
    plt.title("Learning curve of gradient descent")
    plt.xlabel("Number of iterations")
    plt.ylabel("Percentage of correct classifications")
    plt.legend(loc = 'center')
    plt.savefig("part4_learningcurve.png")
    
    plt.figure()
    plt.plot(accuracies[0], accuracies[1], 'g-', label = 'cost function')
    plt.title("Cost function values using gradient descent")
    plt.xlabel("Number of iterations")
    plt.ylabel("f(x)")
    plt.legend(loc = 'upper right')
    plt.savefig("part4_costfunction.png")

    # save weights (includes bias)
    for z in range(10):
        zero = w[z] + b[z]

        im = zero.reshape((28,28))

        imsave('{}.jpg'.format(z), im)
  

## PART5
def f_lin(x, y, theta):
    # cost function
    x = vstack( (ones((1, x.shape[1])), x))
    return sum( (dot(theta.T,x).T - y) ** 2)

def df_lin(x, y, theta):
    # derivative of cost function
    x = vstack( (ones((1, x.shape[1])), x))
    return 2*dot(x, ((dot(theta.T, x).T-y)))

def f_log(x, y_, w):
    # cost function y_ * log(softmax(y))
    p = calculate_output(w, x)
    return cross_entropy(p, y_)

def df_log(x, y_, w):    
    p = calculate_output(w, x)
    x = vstack( (ones((1, x.shape[1])), x))
    return dot(x,(p-y_).T)
    
def grad_descent_lin(cost, grad, x, y, init_t, alpha):
    EPS = 1e-5   #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = 500
    iter  = 0
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha*grad(x, y, t)
        if iter % 25 == 0:
            print ("Iter", iter)
            print("Lin f(x) = ", cost(x, y, t))
        iter += 1
    return t

def grad_descent_log(cost, grad, x, y, init_t, alpha):
    EPS = 1e-5   # EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = 500
    iter = 0
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha*grad(x, y, t)         
        if iter % 25 == 0:
            print ("Iter", iter)
            c = cost(x, y, t)   
            print ("Log f(x) = ", c) 
        iter += 1
    return t


def part5():
    '''
    when there's outliers in y-labels, linear regression overadjusts due to
    high cost function associated with the outliers. here we generate incorrect
    y-labels (10%) to demonstrate overfitting with linear regression and the 
    better performance of logistic regression 
    
    we use the same images of digits as before, but mislabel 10% of the y-values
    this reduces performance using linear regression but affects logistic regression
    much less
    '''
    
    np.random.seed(2345)
    
    n = 500
    x = M["train0"][0:n]
    x = vstack((x, M["train1"][0:n]))
    x = vstack((x, M["train2"][0:n]))
    x = vstack((x, M["train3"][0:n]))
    x = vstack((x, M["train4"][0:n]))
    x = vstack((x, M["train5"][0:n]))
    x = vstack((x, M["train6"][0:n]))
    x = vstack((x, M["train7"][0:n]))
    x = vstack((x, M["train8"][0:n]))
    x = vstack((x, M["train9"][0:n]))
    x = x.T/255.0

    nT = n//10
    xT = M["test0"][0:nT]
    xT = vstack((xT, M["test1"][0:nT]))
    xT = vstack((xT, M["test2"][0:nT]))
    xT = vstack((xT, M["test3"][0:nT]))
    xT = vstack((xT, M["test4"][0:nT]))
    xT = vstack((xT, M["test5"][0:nT]))
    xT = vstack((xT, M["test6"][0:nT]))
    xT = vstack((xT, M["test7"][0:nT]))
    xT = vstack((xT, M["test8"][0:nT]))
    xT = vstack((xT, M["test9"][0:nT]))
    xT = xT.T/255.0
    
    # onehot encoding 
    y_ = []    
    for i in range(0, n):
        y_.append([1,0,0,0,0,0,0,0,0,0])        
    for i in range(0, n):
        y_.append([0,1,0,0,0,0,0,0,0,0])        
    for i in range(0, n):
        y_.append([0,0,1,0,0,0,0,0,0,0])        
    for i in range(0, n):
        y_.append([0,0,0,1,0,0,0,0,0,0])        
    for i in range(0, n):
        y_.append([0,0,0,0,1,0,0,0,0,0])        
    for i in range(0, n):
        y_.append([0,0,0,0,0,1,0,0,0,0])        
    for i in range(0, n):
        y_.append([0,0,0,0,0,0,1,0,0,0])        
    for i in range(0, n):
        y_.append([0,0,0,0,0,0,0,1,0,0])        
    for i in range(0, n):
        y_.append([0,0,0,0,0,0,0,0,1,0])         
    for i in range(0, n):
        y_.append([0,0,0,0,0,0,0,0,0,1])
        
    yT = []    
    for i in range(0, nT):
        yT.append([1,0,0,0,0,0,0,0,0,0])        
    for i in range(0, nT):
        yT.append([0,1,0,0,0,0,0,0,0,0])        
    for i in range(0, nT):
        yT.append([0,0,1,0,0,0,0,0,0,0])        
    for i in range(0, nT):
        yT.append([0,0,0,1,0,0,0,0,0,0])        
    for i in range(0, nT):
        yT.append([0,0,0,0,1,0,0,0,0,0])        
    for i in range(0, nT):
        yT.append([0,0,0,0,0,1,0,0,0,0])        
    for i in range(0, nT):
        yT.append([0,0,0,0,0,0,1,0,0,0])        
    for i in range(0, nT):
        yT.append([0,0,0,0,0,0,0,1,0,0])        
    for i in range(0, nT):
        yT.append([0,0,0,0,0,0,0,0,1,0])         
    for i in range(0, nT):
        yT.append([0,0,0,0,0,0,0,0,0,1])
        
    yT = array(yT)
        
    # generate some mislabels
    for i in range(nT):
        # each onehot y value is shifted over to the next digit
        y_[i] = [0,1,0,0,0,0,0,0,0,0]
        y_[i+500] = [0,0,1,0,0,0,0,0,0,0]
        y_[i+1000] = [0,0,0,1,0,0,0,0,0,0]
        y_[i+1500] = [0,0,0,0,1,0,0,0,0,0]
        y_[i+2000] = [0,0,0,0,0,1,0,0,0,0]
        y_[i+2500] = [0,0,0,0,0,0,1,0,0,0]
        y_[i+3000] = [0,0,0,0,0,0,0,1,0,0]
        y_[i+3500] = [0,0,0,0,0,0,0,0,1,0]
        y_[i+4000] = [0,0,0,0,0,0,0,0,0,1]
        y_[i+4500] = [0,0,0,0,0,0,0,0,0,0]
        
    y_ = array(y_).T
    
    # initial weights, bias is w[0]
    w_init = random.normal(0, 0.001, (785, 10))
    
    # learning step
    delta = 0.000005
    w_log = grad_descent_log(f_log, df_log, x, y_, w_init, delta)
    w_lin = grad_descent_lin(f_lin, df_lin, x, y_.T, w_init, delta)
    
    # check accuracy of log
    p = calculate_output(w_log, xT).T
    correct = 0
    incorrect = 0;
    for i in range(500):
        if (argmax(p[i]) == argmax(yT[i])):
            correct += 1
        else:
            incorrect += 1
    print("Logistic, ", correct / (correct + incorrect))
    
    # check accuracy of linear
    p = dot(w_lin[1:].T, xT).T + w_lin[0]
    correct = 0.
    incorrect = 0.
    for i in range(500):
        if (argmax(p[i]) == argmax(yT[i])):
            correct += 1
        else:
            incorrect += 1
    print("Linear: ", correct / (correct + incorrect))
    
    print("Lin f(x): ", f_lin(x, y_.T, w_lin))    
    print("Log f(x): ", f_log(x, y_, w_log))
    
## uncomment each part to run it individually
# 
# part1()
# part2()
# part3()
# part4()
# part5()

