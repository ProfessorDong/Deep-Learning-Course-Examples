import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.cm as cm

#Read training and test data from MNIST dataset 
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0
x_train = np.float32(x_train)

im = x_train[100]
#im = im.reshape(28,28)

plt.imshow(im, cmap = cm.Greys)
print(y_train[100])

#n_input = 784  # MNIST image shape 28*28
n_classes = 10   # MNIST classes 0-9 ten digits

x = tf.placeholder(tf.float32, [None, 28, 28])

im = tf.reshape(x, shape=[-1, 28, 28, 1])  # Reshape for Conv2D

#W = tf.Variable(tf.zeros([n_input, n_classes]))
#b = tf.Variable(tf.zeros([n_classes]))

y_expected = tf.placeholder(tf.int32)

# First Convolutional Layer
# 6 convolutional 5x5 filters and biases on the first layer.
with tf.variable_scope("FirstConvLayer"):
    F1 = tf.Variable(tf.random_normal([5, 5, 1, 6]))
    b1 = tf.Variable(tf.random_normal([6]))
    F1_im = tf.nn.conv2d(im, F1, strides=[1,1,1,1], padding='SAME')
    h1 = tf.nn.tanh(tf.nn.bias_add(F1_im, b1))

# First Pooling Layer
# Pooling on 2x2 regions
with tf.variable_scope("FirstPoolingLayer"):
    h2 = tf.nn.avg_pool(h1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
# Second Convolutional Layer
# 16 convolutional 5x5x32 filters and biases on the first layer.
with tf.variable_scope("SecondConvLayer"):
    F3 = tf.Variable(tf.random_normal([5, 5, 6, 16]))
    b3 = tf.Variable(tf.random_normal([16]))
    F3_im = tf.nn.conv2d(h2, F3, strides=[1,1,1,1], padding='VALID')
    h3 = tf.nn.tanh(tf.nn.bias_add(F3_im, b3))

# Second Pooling Layer
# Pooling on 2x2 regions
with tf.variable_scope("SecondPoolingLayer"):
    h4 = tf.nn.avg_pool(h3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')   

# Three Fully Connected Layers
# First Fully Connected Layer, 120 outputs
with tf.variable_scope("FirstFullyConnected"):
    h4_vect = tf.reshape(h4, [-1, 5*5*16])
    W5 = tf.Variable(tf.random_normal([5*5*16, 120]))
    b5 = tf.Variable(tf.random_normal([120]))
    h5 = tf.nn.relu(tf.add(tf.matmul(h4_vect, W5),b5))

# Second Fully Connected Layer, 120 inputs, 84 outputs
with tf.variable_scope("SecondFullyConnected"):
    W6 = tf.Variable(tf.random_normal([120, 84]))
    b6 = tf.Variable(tf.random_normal([84]))
    h6 = tf.nn.relu(tf.add(tf.matmul(h5, W6),b6))

# Third Fully Connected Layer, 84 inputs, 10 outputs
with tf.variable_scope("ThirdFullyConnected"):
    W7 = tf.Variable(tf.random_normal([84, n_classes]))
    b7 = tf.Variable(tf.random_normal([n_classes]))
    y_pred = tf.add(tf.matmul(h6,W7), b7)
    
#y_pred = tf.nn.softmax(tf.add(tf.matmul(x,W), b))

# Loss function - Cross Entropy
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_exp * tf.log(y_pred),reduction_indices=[1]))
Loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_expected, logits=y_pred))

optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(Loss)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()


# Optimization
step = 1
training_iters = 1000
batch_size = 128

while step < training_iters:
    idx = np.random.permutation(x_train.shape[0])[:batch_size]
    feed_dict = {x:np.float32(x_train[idx]),y_expected:y_train[idx]}
    sess.run(optimizer, feed_dict=feed_dict)
    step += 1

#File_Writer = tf.summary.FileWriter('.\\LeNet_MNIST_graph',sess.graph)

# Evaluation
Is_prediction_correct = tf.equal(tf.cast(tf.argmax(y_pred,1), tf.int32), y_expected)
accuracy = tf.reduce_mean(tf.cast(Is_prediction_correct, tf.float32))

accu = sess.run([accuracy], feed_dict={x:np.float32(x_test), y_expected:y_test})

print(accu)

sess.close()
