# MNIST - training and testing numbers 0-9
# @param nodes_hl1, hl2, hl3 - nodes in hidden layer 1
# @param label - defines what number it is, hot one encoded (number 0-9)
# @param batch_size - size of training dataset in one iteration
# @param x,y - placeholders for image and labels
# @param initializer - using xavier initializer
# @param hidden_layer_1,2,3 - hidden layers in neural_network
# @param l1, l2, l3 - (data * weights) + biases
# @param optimizer - AdamOptimizer
# @param cost - measures the probability of error
# @param correct - checks the predicted value to original value
# @param accuracy - accuracy of the neural network
# @param num_epochs - how many times the whole dataset is trained

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot = True)

nodes_hl1 = 512
nodes_hl2 = 256
nodes_hl3 = 128

label = 10
batch_size = 100

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float', [None, 10])

initializer = tf.contrib.layers.xavier_initializer()

hidden_layer_1 = {'weights':tf.Variable(initializer([784, nodes_hl1])),
                  'biases':tf.Variable(initializer([nodes_hl1]))}
hidden_layer_2 = {'weights':tf.Variable(initializer([nodes_hl1, nodes_hl2])),
                  'biases':tf.Variable(initializer([nodes_hl2]))}
hidden_layer_3 = {'weights':tf.Variable(initializer([nodes_hl2, nodes_hl3])),
                  'biases':tf.Variable(initializer([nodes_hl3]))}
output_layer = {'weights':tf.Variable(initializer([nodes_hl3, label])),
                  'biases':tf.Variable(initializer([label]))}
#creating model
def neural_network_model(data):

    l1 = tf.add(tf.matmul(data, hidden_layer_1['weights']), hidden_layer_1['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']), hidden_layer_2['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_layer_3['weights']), hidden_layer_3['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output

#training the model
def train_neural_network(x):
  prediction = neural_network_model(x)
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))
  optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

  correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
  accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
  num_epochs = 10
  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for epoch in range(num_epochs):
      epoch_loss = 0
      for _ in range(int(mnist.train.num_examples/batch_size)):
        epoch_x,epoch_y = mnist.train.next_batch(batch_size)
        _,c = sess.run([optimizer,cost], feed_dict = {x:epoch_x, y:epoch_y})
        epoch_loss += c

      print('Epoch', epoch + 1, 'completed out of', num_epochs, ' with loss:', epoch_loss , 'Training accuracy:', accuracy.eval({x:epoch_x, y:epoch_y}))

    print('Accuracy:' , accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

    # to save weights and biases

    # array = hidden_layer_1['weights'].eval(sess)
    # np.savetxt("W1.csv", array, delimiter=",")
    #
    # array = hidden_layer_1['biases'].eval(sess)
    # np.savetxt("B1.csv", array, delimiter=",")
    #
    # array = hidden_layer_2['weights'].eval(sess)
    # np.savetxt("W2.csv", array, delimiter=",")
    #
    # array = hidden_layer_2['biases'].eval(sess)
    # np.savetxt("B2.csv", array, delimiter=",")
    #
    # array = hidden_layer_3['weights'].eval(sess)
    # np.savetxt("W3.csv", array, delimiter=",")
    #
    # array = hidden_layer_2['biases'].eval(sess)
    # np.savetxt("B3.csv", array, delimiter=",")
    #
    # array = output_layer['weights'].eval(sess)
    # np.savetxt("WO.csv", array, delimiter=",")
    #
    # array = output_layer['biases'].eval(sess)
    # np.savetxt("BO.csv", array, delimiter=",")

train_neural_network(x)
