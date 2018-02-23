################################################################################
#Michael Guerzhoy and Davi Frossard, 2016
#AlexNet implementation in TensorFlow, with weights
#Details: 
#http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
#With code from https://github.com/ethereon/caffe-tensorflow
#Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
#Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
#
#
################################################################################

from numpy import *
import os
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
from numpy import random
import get_data_dictionary
from download_images import download_images

import hashlib
import scipy
from PIL import Image
import urllib.request

import tensorflow as tf



act = ['drescher', 'ferrera', 'chenoweth', 'baldwin', 'hader', 'carell']

## Download images if the folders do not exist yet (crude but works)

flag = False

if not os.path.exists("actors"):
    os.mkdir("actors")
    flag = True
if not os.path.exists("actresses"):
    os.mkdir("actresses")
    flag = True
if not os.path.exists("cropped_actors"):
    os.mkdir("cropped_actors")
    flag = True
if not os.path.exists("cropped_actresses"):
    os.mkdir("cropped_actresses")
    flag = True

if flag:   
    download_images()
    
## Reuse part7 code to get y array


M = get_data_dictionary.get_data_dictionary()


## Fetch data using new 227, 227, 3 size images

def get_data():
    training = []
    validation = []
    test = []
    scrubs = ['baldwin', 'hader', 'carell']
    scrubesses = ['drescher', 'ferrera', 'chenoweth']
    for name in scrubs:
        for i in range(90):
            im = (imread("actors/" + name + str(i) + ".jpg")).astype(float32)
            im = im - mean(im)
            im[:, :, 0], im[:, :, 2] = im[:, :, 2], im[:, :, 0]
            training.append(im)
        for i in range(90, 100):
            im = (imread("actors/" + name + str(i) + ".jpg")).astype(float32)
            im = im - mean(im)
            im[:, :, 0], im[:, :, 2] = im[:, :, 2], im[:, :, 0]
            validation.append(im)
        for i in range(100, 130):
            im = (imread("actors/" + name + str(i) + ".jpg")).astype(float32)
            im = im - mean(im)
            im[:, :, 0], im[:, :, 2] = im[:, :, 2], im[:, :, 0]
            test.append(im)
    
    for name in scrubesses:
        for i in range(90):
            im = (imread("actresses/" + name + str(i) + ".jpg")).astype(float32)
            im = im - mean(im)
            im[:, :, 0], im[:, :, 2] = im[:, :, 2], im[:, :, 0]
            training.append(im)
        for i in range(90, 100):
            im = (imread("actresses/" + name + str(i) + ".jpg")).astype(float32)
            im = im - mean(im)
            im[:, :, 0], im[:, :, 2] = im[:, :, 2], im[:, :, 0]
            validation.append(im)
        for i in range(100, 130):
            im = (imread("actresses/" + name + str(i) + ".jpg")).astype(float32)
            im = im - mean(im)
            im[:, :, 0], im[:, :, 2] = im[:, :, 2], im[:, :, 0]
            test.append(im)
    
    return training, validation, test
    

def get_test(M):
    batch_xs = zeros((0, 32*32))
    batch_y_s = zeros((0, 6))

    test_k =  ["test"+str(name) for name in act]
    for k in range(6):
        batch_xs = vstack((batch_xs, ((array(M[test_k[k]])[:])/255.)  ))
        one_hot = zeros(6)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M[test_k[k]]), 1))   ))
    return batch_xs, batch_y_s
    
def get_train(M):
    batch_xs = zeros((0, 32*32))
    batch_y_s = zeros( (0, 6))

    train_k =  ["train"+str(name) for name in act]
    for k in range(6):
        batch_xs = vstack((batch_xs, ((array(M[train_k[k]])[:])/255.)  ))
        one_hot = zeros(6)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M[train_k[k]]), 1))   ))
    return batch_xs, batch_y_s
    
def get_valid(M):
    batch_xs = zeros((0, 32*32))
    batch_y_s = zeros( (0, 6))

    train_k =  ["valid"+str(name) for name in act]
    for k in range(6):
        batch_xs = vstack((batch_xs, ((array(M[train_k[k]])[:])/255.)  ))
        one_hot = zeros(6)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M[train_k[k]]), 1))   ))
    return batch_xs, batch_y_s
    
x_train, x_valid, x_test = get_data()

## AlexNet Implementation up to Conv4

net_data = load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups =  tf.split(input, group, 3)   #tf.split(3, group, input)
        kernel_groups = tf.split(kernel, group, 3)  #tf.split(3, group, kernel) 
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)          #tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])



x = tf.placeholder(tf.float32, (None,) + (227, 227, 3))


#conv1
#conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
conv1W = tf.Variable(net_data["conv1"][0])
conv1b = tf.Variable(net_data["conv1"][1])
conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
conv1 = tf.nn.relu(conv1_in)

#lrn1
#lrn(2, 2e-05, 0.75, name='norm1')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool1
#max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


#conv2
#conv(5, 5, 256, 1, 1, group=2, name='conv2')
k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
conv2W = tf.Variable(net_data["conv2"][0])
conv2b = tf.Variable(net_data["conv2"][1])
conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv2 = tf.nn.relu(conv2_in)


#lrn2
#lrn(2, 2e-05, 0.75, name='norm2')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool2
#max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#conv3
#conv(3, 3, 384, 1, 1, name='conv3')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
conv3W = tf.Variable(net_data["conv3"][0])
conv3b = tf.Variable(net_data["conv3"][1])
conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv3 = tf.nn.relu(conv3_in)

#conv4
#conv(3, 3, 384, 1, 1, group=2, name='conv4')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
conv4W = tf.Variable(net_data["conv4"][0])
conv4b = tf.Variable(net_data["conv4"][1])
conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv4 = tf.nn.relu(conv4_in)

## Part 10

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# extract activations for conv4
train_cv4 = sess.run(conv4, feed_dict={x:x_train})
valid_cv4 = sess.run(conv4, feed_dict={x:x_valid})
test_cv4 = sess.run(conv4, feed_dict={x:x_test})
cv4_out = tf.placeholder(tf.float32, (None, 13, 13, 384))


# fully connected nn using conv4 as inputs
seed = 1234
nhid = 30
W0 = tf.Variable(tf.truncated_normal((13 * 13 * 384,)+(nhid,), stddev=0.01, seed=seed))
b0 = tf.Variable(tf.truncated_normal([nhid], stddev=0.01, seed=seed))

W1 = tf.Variable(tf.truncated_normal([nhid, 6], stddev=0.01, seed=seed))
b1 = tf.Variable(tf.truncated_normal([6], stddev=0.01, seed=seed))

layer1 = tf.nn.relu(tf.matmul(tf.reshape(cv4_out, [-1, 13 * 13 * 384]), W0)+b0)
layer2 = tf.matmul(layer1, W1)+b1

y = tf.nn.softmax(layer2)
y_ = tf.placeholder(tf.float32, [None, 6])

lam = 0.0001
decay_penalty =lam*tf.reduce_sum(tf.square(W0))+lam*tf.reduce_sum(tf.square(W1))
reg_NLL = -tf.reduce_sum(y_*tf.log(y))+decay_penalty

train_step = tf.train.AdamOptimizer(0.0004).minimize(reg_NLL)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# get data using the master dictionary (only the ys are used for comparison)
xt, yt = get_train(M)
test_x, test_y = get_test(M)
xv, yv = get_valid(M)

learning_curve = []
t = time.time()

for i in range(300):
    sess.run(train_step, feed_dict={cv4_out: train_cv4, y_: yt})

    if i % 5 == 0:
        print("i=",i)
        test = sess.run(accuracy, feed_dict={cv4_out: test_cv4, y_: test_y})
        valid = sess.run(accuracy, feed_dict={cv4_out: valid_cv4, y_: yv})
        train = sess.run(accuracy, feed_dict={cv4_out: train_cv4, y_: yt})
        print("Train:", train)
        print("Validation:", valid)
        print("Test:", test)
        print("Penalty:", sess.run(decay_penalty))
        learning_curve.append([i, train, valid, test])

learning_curve = array(learning_curve).T

# plot and save learning curves
plt.figure()
plt.plot(learning_curve[0], learning_curve[1] * 100, 'b-', label = 'training set performance')
plt.plot(learning_curve[0], learning_curve[2] * 100, 'g-', label = 'validation set performance')
plt.plot(learning_curve[0], learning_curve[3] * 100, 'r-', label = 'test set performance')
plt.title("Learning curve of AlexNet using AdamOptimizer")
plt.xlabel("Number of iterations")
plt.ylabel("Percentage of correct classifications")
plt.legend(loc = 'center')
plt.savefig("part10_learningcurve.png")

print(time.time()-t)

## Part 11

if not os.path.exists("hidden_units"):
    os.mkdir("hidden_units")
os.chdir("hidden_units")

weights = W0.eval(session=sess)
bias = b0.eval(session=sess)
w = weights.reshape((13, 13, 384, 30))

for i in range(1):
    for j in range(384):
        im = w[:,:,j:(j+1),i:(i+1)]
        im = np.squeeze(im)
        im = im + bias[i]
        imsave("weight" + str(j) + "neuron" + str(i) + ".jpg", im)


units = sess.run(conv4, feed_dict={x: x_train[0:1]})

for i in range(384):
    im = units[:,:,:,i:(i+1)]
    im = np.squeeze(im)
    imsave("feature" + str(i) + ".jpg", im)
    
os.chdir("..")
