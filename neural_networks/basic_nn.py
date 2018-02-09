# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 17:02:05 2018

@author: K.Ataman
"""

import tensorflow as tf
import numpy as np

num_neuron_list = [256, 256, 256]
num_hidden_layers = len(num_neuron_list)

num_entries = 1024
num_features_input = 4
num_features_output = 2

training_epochs = 50
testing_epochs  = 10
# the // ensures the output is an intager, and not a float
mini_batch_size = num_entries // 8
learning_rate = 0.01

def create_dataset(num_entries, num_features_input, num_features_output):
    x = np.random.rand(num_entries, num_features_input)
    y = np.zeros([num_entries,num_features_output])
    y_divisor = 1 / num_features_output
    for i in range(num_entries):
        y_entry = 0
        for j in range(num_features_input):
            if j%2==0:
                y_entry += x[i, j]
            else:
                y_entry = y_entry*x[i, j]
        for j in range(1, num_features_output + 1):
            if y_entry < j * y_divisor:
                y[i, j - 1] = 0
            else:
                y[i, j - 1] = 1
    return [x,y]

def get_batch(batch, batch_size, x, y):
    batch = batch
    batch_size = batch_size
    x = x[batch*batch_size : (batch + 1) * batch_size]
    y = y[batch*batch_size : (batch + 1) * batch_size]
    return [x,y]

x = tf.placeholder("float", [mini_batch_size, num_features_input])
y = tf.placeholder("float", [mini_batch_size, num_features_output])

#weight and bias matrices
w = []
b = []
for i in range(num_hidden_layers):
    #input gate
    if i==0:
        w.append(tf.Variable(tf.random_normal([num_features_input, num_neuron_list[0]])))
        b.append(tf.Variable(tf.random_normal([num_neuron_list[0]])))
    else:
        w.append(tf.Variable(tf.random_normal([num_neuron_list[i - 1], num_neuron_list[i]])))
        b.append(tf.Variable(tf.random_normal([num_neuron_list[i]])))
        
#output gate
w.append(tf.Variable(tf.random_normal([num_neuron_list[num_hidden_layers - 1], num_features_output])))
b.append(tf.Variable(tf.random_normal([num_features_output])))

for i in range(num_hidden_layers + 1):
    if i==0:
        output = tf.nn.sigmoid(tf.add(tf.matmul(x, w[0]), b[0]))
    elif i < num_hidden_layers + 1:
        output = tf.nn.sigmoid(tf.add(tf.matmul(output, w[i]), b[i]))
    #last layer
    else:
        output = tf.add(tf.matmul(output, w[i]), b[i])
        
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = output, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
accuracy, accuracy_update = tf.metrics.accuracy(tf.argmax(output), tf.argmax(y))

#correct = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
#correct = tf.equal(output, y)
#accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

training_cost_list = []
training_accuracy_list = []
testing_cost_list = []
testing_accuracy_list = []

init_g = tf.global_variables_initializer()
#metrics initialization
init_l = tf.local_variables_initializer()
with tf.Session() as sess:
    sess.run(init_g)
    sess.run(init_l)
    
    #training
    for epoch in range(training_epochs):
        avg_cost = 0
        avg_accuracy = 0
        #create new dataset for each epoch to avoid overfitting
        x_, y_ = create_dataset(num_entries, num_features_input, num_features_output)
        for batch in range(int(num_entries // mini_batch_size)):
            batch_x, batch_y = get_batch(batch, mini_batch_size, x_ , y_)
            #Notice how "optimizer", which is the variable we want, is the first entry
            sess.run(optimizer, feed_dict = {x: batch_x, y: batch_y})
            avg_cost += sess.run(cost, feed_dict = {x: batch_x, y: batch_y}) / mini_batch_size
            avg_accuracy += sess.run(accuracy_update, feed_dict = {x: batch_x, y: batch_y}) / mini_batch_size
        print ("Training epoch: ", '%04d' % (epoch+1), "cost=","{:.9f}".format(avg_cost))
        print("Training epoch: ", '%04d' % (epoch+1),"accuracy=","{:.9f}".format(avg_accuracy))
        training_cost_list.append(avg_cost)
        training_accuracy_list.append(avg_accuracy)
    
    #testing
    for epoch in range(testing_epochs):
        avg_cost = 0
        avg_accuracy = 0
        #create new dataset for each epoch to avoid overfitting
        x_, y_ = create_dataset(num_entries, num_features_input, num_features_output)
        for batch in range(int(num_entries // mini_batch_size)):
            batch_x, batch_y = get_batch(batch, mini_batch_size, x_ , y_)
            #Notice how "optimizer", which is the variable we want, is the first entry
            avg_cost += sess.run(cost, feed_dict = {x: batch_x, y: batch_y}) / mini_batch_size
            avg_accuracy += sess.run(accuracy_update, feed_dict = {x: batch_x, y: batch_y}) / mini_batch_size
        print ("Testing epoch: ", '%04d' % (epoch+1), "cost=","{:.9f}".format(avg_cost))
        print("Testing epoch: ", '%04d' % (epoch+1),"accuracy=","{:.9f}".format(avg_accuracy))
        testing_cost_list.append(avg_cost)
        testing_accuracy_list.append(avg_accuracy)