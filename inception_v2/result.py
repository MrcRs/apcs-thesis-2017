import tensorflow as tf
import numpy as np
import sys

output_path = 'output/2_faces/'

num_classes = 8

x = tf.placeholder(tf.float32, [None, 37])
y_true = tf.placeholder(tf.float32, [None, num_classes])

weights = tf.Variable(tf.zeros([37, num_classes]))
biases = tf.Variable(tf.zeros([num_classes]))

logits = tf.matmul(x, weights) + biases

y_pred = tf.nn.softmax(logits)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)

correct_prediction = tf.diag_part(tf.matmul(y_pred, tf.transpose(y_true)))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()
session.run(tf.global_variables_initializer())

def optimize(num_iterations):
	x_batch = np.load(output_path + 'softmax_vector.npy')
	y_true_batch = np.load(output_path + 'relation_traits_vector.npy')
	feed_dict = {x: x_batch, y_true: y_true_batch}
	for i in range(num_iterations):
		session.run(optimizer, feed_dict=feed_dict)
		sys.stdout.write("\rStep: {0}".format(i))
		sys.stdout.flush()
	print()

def print_accuracy():
	x_batch = np.load(output_path + 'softmax_vector_test.npy')
	y_true_batch = np.load(output_path + 'relation_traits_vector_test.npy')
	feed_dict = {x: x_batch, y_true: y_true_batch}
	acc = session.run(accuracy, feed_dict=feed_dict)
	print("Accuracy on test-set: {0:.1%}".format(acc))

print_accuracy()
optimize(50000)
print_accuracy()

saver = tf.train.Saver()
saver.save(session, output_path + '2_faces')