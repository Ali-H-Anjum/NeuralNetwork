import numpy as np
import tensorflow as tf
from keras.datasets import mnist

import seaborn as sns
import matplotlib.pyplot as plt

tf.compat.v1.disable_v2_behavior()

def weight_variable(name, shape):
    he_normal = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None) 
    return tf.Variable(he_normal(shape=shape), name=name)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#parameters
learning_rate = 0.001
training_epochs = 1000
batch_size = 50
display_step = 100



x = tf.compat.v1.placeholder(tf.float32, [None, 784])
y_= tf.compat.v1.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1,28,28,1])

#layer 1
W_conv1 = weight_variable("W1", [5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#layer 2
W_conv2 = weight_variable("W2", [5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#layer 3
W_fc1 = weight_variable("W3", [7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#dropout
keep_prob = tf.compat.v1.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#layer 5
W_fc2 = weight_variable("W5", [1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

#optimizers
train_step1 = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
train_step2 = tf.compat.v1.train.MomentumOptimizer(learning_rate, momentum=0.9, use_nesterov=True).minimize(cross_entropy)
train_step3 = tf.compat.v1.train.RMSPropOptimizer(learning_rate).minimize(cross_entropy)
train_step4 = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        avg_cost = 0.0
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch = mnist.train.next_batch(batch_size)
            _, c = sess.run([train_step1, cross_entropy], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            avg_cost += c/total_batch
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
