import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

sess = tf.InteractiveSession()

mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

# initial parameters
width = 28
height = 28
flat = width * height
class_output = 10

# input and output 
x = tf.placeholder(tf.float32, shape = [None, flat])
y_ = tf.placeholder(tf.float32, shape= [None, class_output])

# converting images to tensor
x_image = tf.reshape(x, [-1,28,28,1])

# First Convolutional Layer
W_conv1 = tf.Variable(tf.truncated_normal([5,5,1,32], stddev = 0.1)) # Kernel of size 5x5 and input channel 1 with 32 features maps
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32])) # for 32 outputs

convolve1 = tf.nn.conv2d(x_image, W_conv1, strides=[1,1,1,1], padding='SAME') + b_conv1
h_conv1 = tf.nn.relu(convolve1) # relu activation function
conv1 = tf.nn.max_pool(h_conv1, ksize = [1,2,2,1], strides = [1,2,2,1], padding= 'SAME') # maxpool 2x2 with no overlapping
''' the output here is a image of dimension 14x14x32 '''

# Second Convolutional Layer
W_conv2 = tf.Variable(tf.truncated_normal([5,5,32,64], stddev = 0.1)) # Kernel size 5x5 and input channel 32 with 64 features maps
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64])) # for 64 outputs

convolve2 = tf.nn.conv2d(conv1, W_conv2, strides = [1,1,1,1], padding= 'SAME') + b_conv2
h_conv2 = tf.nn.relu(convolve2)
conv2 = tf.nn.max_pool(h_conv2, ksize= [1,2,2,1], strides = [1,2,2,1], padding = 'SAME') # maxpool 2x2 
''' the output here is 7x7x64 '''

# Fully Connected Layer
flattening_matrix = tf.reshape(conv2, [-1,7*7*64]) # Flattening the second convolution layer

W_fc1 = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1)) # fully connected layer weights
b_fc1 = tf.Variable(tf.constant(0.1, shape = [1024])) # need 1024 biases for 1024 outputs

fully_connected_layer = tf.matmul(flattening_matrix, W_fc1) + b_fc1

h_fc1 = tf.nn.relu(fully_connected_layer)

# To reduce overfitting
keep_prob = tf.placeholder(tf.float32)
layer_drop = tf.nn.dropout(h_fc1, keep_prob)


# Softmax layer
W_fc2 = tf.Variable(tf.truncated_normal([1024,10], stddev = 0.1)) # 1024 neurons
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10])) # for 10 outputs

fc = tf.matmul(layer_drop, W_fc2) + b_fc2

y_cnn = tf.nn.softmax(fc)


