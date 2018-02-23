from pylab import *
import time
from numpy import random
from download_images import download_images
from get_data_dictionary import get_data_dictionary
import os

import pickle as cPickle

from scipy.io import loadmat

import tensorflow as tf


t = int(time.time())
# t = 1454219613
print("t=", t)
random.seed(t)

################################################################################

# uncomment to create folders and download images

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
    
################################################################################

M = get_data_dictionary()
act = ['drescher', 'ferrera', 'chenoweth', 'baldwin', 'hader', 'carell']

def get_train_batch(M, N):
    n = N//6
    batch_xs = zeros((0, 32*32))
    batch_y_s = zeros((0, 6))

    train_k = ["train"+str(name) for name in act]

    for k in range(6):
        train_size = len(M[train_k[k]])
        idx = array(random.permutation(train_size)[:n])
        batch_xs = vstack((batch_xs, ((array(M[train_k[k]])[idx])/255.)  ))
        one_hot = zeros(6)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (n, 1))   ))
    return batch_xs, batch_y_s


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


def get_valid(M):
    batch_xs = zeros((0, 32*32))
    batch_y_s = zeros((0, 6))

    test_k =  ["valid"+str(name) for name in act]
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


if __name__ == "__main__":
    ## Part 7:
    # placeholder input for x, specifying type = float32, shape = [any, 784]
    x = tf.placeholder(tf.float32, [None, 1024])

    seed = 1234
    nhid = 300
    # initialize weights and bias using truncated normal
    W0 = tf.Variable(tf.truncated_normal([1024, nhid], stddev=0.01, seed=seed))
    b0 = tf.Variable(tf.truncated_normal([nhid], stddev=0.01, seed=seed))

    W1 = tf.Variable(tf.truncated_normal([nhid, 6], stddev=0.01, seed=seed))
    b1 = tf.Variable(tf.truncated_normal([6], stddev=0.01, seed=seed))

    layer1 = tf.nn.tanh(tf.matmul(x, W0)+b0)
    layer2 = tf.matmul(layer1, W1)+b1

    y = tf.nn.softmax(layer2)
    y_ = tf.placeholder(tf.float32, [None, 6])

    # weight penalty L2
    lam = 0.0001
    decay_penalty =lam*tf.reduce_sum(tf.square(W0))+lam*tf.reduce_sum(tf.square(W1))
    reg_NLL = -tf.reduce_sum(y_*tf.log(y))+decay_penalty

    train_step = tf.train.AdamOptimizer(0.0004).minimize(reg_NLL)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # full test and validation sets, 10 in validation, 30 in test
    test_x, test_y = get_test(M)
    valid_x, valid_y = get_valid(M)

    learning_curve = []

    for i in range(1851):
        batch_xs, batch_ys = get_train_batch(M, 60)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        if i % 25 == 0:
            print("i=",i)
            test_accuracy = sess.run(accuracy, feed_dict={x: test_x, y_: test_y})
            valid_accuracy = sess.run(accuracy, feed_dict={x: valid_x, y_: valid_y})
            print("Test:", test_accuracy)
            print("Validation:", valid_accuracy)
            
            batch_xs, batch_ys = get_train(M)
            train_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
            
            print("Train:", train_accuracy)
            print("Penalty:", sess.run(decay_penalty))
            
            learning_curve.append([i, train_accuracy, valid_accuracy, test_accuracy])
    
    learning_curve = array(learning_curve).T
    
    # plot and save learning curves
    plt.figure()
    plt.plot(learning_curve[0], learning_curve[1] * 100, 'b-', label = 'training set performance')
    plt.plot(learning_curve[0], learning_curve[2] * 100, 'g-', label = 'validation set performance')
    plt.plot(learning_curve[0], learning_curve[3] * 100, 'r-', label = 'test set performance')
    plt.title("Learning curve of neural network using AdamOptimizer")
    plt.xlabel("Number of iterations")
    plt.ylabel("Percentage of correct classifications")
    plt.legend(loc = 'center')
    plt.savefig("part7_learningcurve.png")
    
    # final classification performance on test set
    test_accuracy = sess.run(accuracy, feed_dict={x: test_x, y_: test_y})
    print("Final performance on test set: ", test_accuracy)
    
    ## Part 9:
    if not os.path.exists("part9"):
        os.mkdir("part9")
    os.chdir("part9")
    
    weights = W0.eval(session=sess)
    bias = b0.eval(session=sess)
    w = weights.reshape((32, 32, 300))
    w_out = W1.eval(session=sess)
    
    # select most influential neuron for each actor
    for i in range(6):
        name = act[i]
        neuron = argmax(w_out.T[i])
        im = w[:,:,neuron] + bias[neuron]
        imsave(name + "neuron" + str(neuron) + ".jpg", im)
        
    os.chdir("..")
    