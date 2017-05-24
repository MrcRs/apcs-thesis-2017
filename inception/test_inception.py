import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import os

import inception
import config

import prettytensor as pt

num_classes=7

model = inception.Inception()

### CALCULATE TRANSFER-VALUES ###

from inception import transfer_values_cache
file_path_cache_test = os.path.join(config.TRANSFER_VALUES_DIR, 'inception_kaggle_test.pkl')
transfer_values_test = transfer_values_cache(cache_path=file_path_cache_test, images=None, model=model)

### NEW CLASSIFIER IN TENSORFLOW ###

# Placeholder Variables
transfer_len = model.transfer_len
x = tf.placeholder(tf.float32, shape=[None, transfer_len], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

# Neural Network
x_pretty = pt.wrap(x)	# Wrap the transfer-values as a Pretty Tensor object.
with pt.defaults_scope(activation_fn=tf.nn.relu):
	y_pred, loss = x_pretty.fully_connected(size=1024, name='layer_fc1').softmax_classifier(num_classes=num_classes, labels=y_true)

# Optimization Method
global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step)

# Classification Accuracy
y_pred_cls = tf.argmax(y_pred, dimension=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

### load test data ###
labels_test = np.load('transfer_values/test_kaggle.npy')
cls_test=np.argmax(labels_test, axis=1)
print(labels_test)

saver = tf.train.Saver()
session = tf.Session()
saver.restore(session, "model/model.ckpt")

# Helper-func. for clculating classifications
batch_size = 256 # Split the data-set in batches of this size to limit RAM usage.

def predict_cls(transfer_values, labels, cls_true):
	# Number of images.
	num_images = len(transfer_values)

	# Allocate an array for the predicted classes which
	# will be calculated in batches and filled into this array.
	cls_pred = np.zeros(shape=num_images, dtype=np.int)

	# Now calculate the predicted classes for the batches.
	# We will just iterate through all the batches.
	# There might be a more clever and Pythonic way of doing this.

	# The starting index for the next batch is denoted i.
	i = 0
	correct = []

	while i < num_images:
		# The ending index for the next batch is denoted j.
		j = min(i + batch_size, num_images)

		# Create a feed-dict with the images and labels
		# between index i and j.
		feed_dict = {x: transfer_values[i:j], y_true: labels[i:j]}

		# Calculate the predicted class using TensorFlow.
		cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
		correct.append(cls_true[i] == cls_pred[i])

		# Set the start-index for the next batch to the
		# end-index of the current batch.
		i = j

	# Create a boolean array whether each image is correctly classified.
	# correct = (cls_true == cls_pred)

	return correct, cls_pred

def predict_cls_test():
	return predict_cls(transfer_values = transfer_values_test, labels = labels_test, cls_true = cls_test)

# Helper-functions for calculating the classification accuracy
def classification_accuracy(correct):
	# When averaging a boolean array, False means 0 and True means 1.
	# So we are calculating: number of True / len(correct) which is
	# the same as the classification accuracy.

	# Return the classification accuracy
	# and the number of correct classifications.

	_sum = 0
	_mean = 0.0
	for i in correct:
		_sum = _sum + i

	_mean = _sum / len(correct)
	return _mean, _sum

	# return correct.mean(), correct.sum()

# Helper-func. for calculating the classification accuracy
def print_test_accuracy(show_example_errors=False, show_confusion_matrix=False):

	# For all the images in the test-set,
	# calculate the predicted classes and whether they are correct.
	correct, cls_pred = predict_cls_test()

	# Classification accuracy and the number of correct classifications.
	acc, num_correct = classification_accuracy(correct)

	# Number of images being classified.
	num_images = len(correct)

	# Print the accuracy.
	msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
	print(msg.format(acc, num_correct, num_images))

	# Plot some examples of mis-classifications, if desired.
	# if show_example_errors:
	# 	print("Example errors:")
	# 	plot_example_errors(cls_pred=cls_pred, correct=correct)

	# Plot the confusion matrix, if desired.
	# if show_confusion_matrix:
	# 	print("Confusion Matrix:")
	# 	plot_confusion_matrix(cls_pred=cls_pred)

print_test_accuracy(show_example_errors=False, show_confusion_matrix=False)