# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 13:12:22 2018

@author: K.Ataman
"""

import tensorflow as tf
import numpy as np

hidden_state_size_list = [256, 256, 256]
dense_state_size = hidden_state_size_list[-1] // 8
num_hidden_layers = len(hidden_state_size_list)

num_entries = 1024
num_features_input = 4
num_features_output = 2
backpropagation = 10

training_epochs = 50
testing_epochs  = 10
# the // ensures the output is an intager, and not a float
mini_batch_size = num_entries // 8
learning_rate = 0.01

x = tf.placeholder("float", [mini_batch_size, backpropagation, num_features_input])
#split allows us to iterate through the tensor
x = tf.split(x, backpropagation, axis = 1)
y = tf.placeholder("float", [mini_batch_size, backpropagation, num_features_output])
y = tf.split(y, backpropagation, axis = 1)

w_in = tf.Variable(tf.random_normal([backpropagation, num_features_input, hidden_state_size_list[0]]))
b_in = tf.Variable(tf.random_normal([backpropagation, mini_batch_size, hidden_state_size_list[0]]))

w = []
b = []
states = []

w.append(w_in)
b.append(b_in)
#what we want: w = [layer, state_size[i], backprop, state_size[i + 1]]
for i in range(num_hidden_layers - 1):
    w_h = tf.Variable(tf.random_normal([backpropagation, hidden_state_size_list[i], hidden_state_size_list[i + 1]]))
    b_h = tf.Variable(tf.random_normal([backpropagation, mini_batch_size, hidden_state_size_list[i + 1]]))
    w.append([w_h])
    b.append([b_h])
    states.append(tf.random_normal[backpropagation, mini_batch_size, hidden_state_size [i]])

w_d = tf.Variable(tf.random_normal([backpropagation, hidden_state_size_list[-1], dense_state_size]))
b_d = tf.Variable(tf.random_normal([backpropagation, mini_batch_size, dense_state_size]))
w.append([w_d])
b.append([b_d])
states.append(tf.random_normal[backpropagation, mini_batch_size, hidden_state_size [-1]])
w_o = tf.Variable(tf.random_normal([backpropagation, dense_state_size, num_features_output]))
b_o = tf.Variable(tf.random_normal([backpropagation, mini_batch_size, num_features_output]))
w.append([w_o])
b.append([b_o])

#actual rnn lstm
for layer in range(len(w)):
    w_current = w[layer]
    b_current = b[layer]
    f_t = []
    i_t = []
    c_t = []
    C-t = []
    o_t = []
    h_t = []
    for time in range(backpropagation):
        if layer!= 0 & layer!= len(w) - 1:
            #state stuff
    
#DONT FORGET THE SPLITS

#f_t = sigm(w*[h,x] +b)
#f_t = tf.nn.sigmoid(tf.add(tf.matmul(output, w[0]), b[0]))