import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import os

import inception
import prettytensor as pt
import config

num_classes = 7

train_batch_size = 64

### DOWNLOAD INCEPTION MODEL ###

inception.maybe_download()

### LOAD INCEPTION MODEL ###

model = inception.Inception()

### CALCULATE TRANSFER-VALUES ###

from inception import transfer_values_cache
kaggle_file_path_cache_train = os.path.join(config.TRANSFER_VALUES_DIR, 'inception_kaggle_train.pkl')
kaggle_transfer_values_train = transfer_values_cache(cache_path=kaggle_file_path_cache_train, images=None, model=model)

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

### LOAD TRUE LABELS ###

kaggle_labels_train = np.load('transfer_values/kaggle.npy')

### TENSORFLOW RUN ###

# Create TensorFlow Session
session = tf.Session()

# Initialize Variables
session.run(tf.global_variables_initializer())

def random_batch(transfer_values_train, labels_train):
	# Number of images (transfer-values) in the training-set.
	num_images = len(transfer_values_train)

	# Create a random index.
	idx = np.random.choice(num_images, size=train_batch_size, replace=False)
	# Use the random index to select random x and y-values.
	# We use the transfer-values instead of images as x-values.
	x_batch = transfer_values_train[idx]
	y_batch = labels_train[idx]

	return x_batch, y_batch

# Helper-func. to perform optimization
def optimize(num_iterations):
	# Start-time used for printing time-usage below.
	start_time = time.time()

	for i in range(num_iterations):
		# Get a batch of training examples.
		# x_batch now holds a batch of images (transfer-values) and
		# y_true_batch are the true labels for those images.
		x_batch, y_true_batch = random_batch(transfer_values_train=kaggle_transfer_values_train, labels_train=kaggle_labels_train)

		# Put the batch into a dict with the proper names
		# for placeholder variables in the TensorFlow graph.
		feed_dict_train = {x: x_batch, y_true: y_true_batch}

		# Run the optimizer using this batch of training data.
		# TensorFlow assigns the variables in feed_dict_train
		# to the placeholder variables and then runs the optimizer.
		# We also want to retrieve the global_step counter.
		i_global, _ = session.run([global_step, optimizer], feed_dict=feed_dict_train)

		# Print status to screen every 100 iterations (and last).
		if (i_global % 100 == 0) or (i == num_iterations - 1):
			# Calculate the accuracy on the training-batch.
			batch_acc = session.run(accuracy, feed_dict=feed_dict_train)

			# Print status.
			msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
			print(msg.format(i_global, batch_acc))

	# Ending time.
	end_time = time.time()

	# Difference between start and end-times.
	time_dif = end_time - start_time

	# Print the time-usage.
	print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

optimize(num_iterations=100000)

saver = tf.train.Saver()
save_path = saver.save(session, "model/model.ckpt")
print("Model save in file :%s" % save_path)

model.close()
session.close()