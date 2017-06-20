import tensorflow as tf
import os
import numpy as np

OUTPUT_DIR = 'output/two_faces'

BATCH_SIZE = 32

TRAIN_SIZE = 7226
TEST_SIZE = 790

x = tf.placeholder(tf.float32, shape=[None, 1536 * 4], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, 8], name='y_true')

weights = tf.Variable(tf.zeros([1536 * 4, 8]))
biases = tf.Variable(tf.zeros([8]))

logits = tf.matmul(x, weights) + biases
y_pred = tf.nn.sigmoid(logits)

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y_true)
cost = tf.reduce_mean(cross_entropy)

global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost, global_step)

correct_prediction = (tf.diag_part(tf.matmul(y_pred, tf.transpose(y_true))) + tf.diag_part(tf.matmul(1 - y_pred, tf.transpose(1 - y_true)))) / 8
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

label_correct_prediction = (tf.diag_part(tf.matmul(tf.transpose(y_pred), y_true)) + tf.diag_part(tf.matmul(tf.transpose(1- y_pred), 1 - y_true))) / TEST_SIZE

# precision = tf.metrics.precision(y_true, tf.round(y_pred))
# recall = tf.metrics.recall(y_true, tf.round(y_pred))

# all_labels_true = tf.reduce_min(tf.cast(correct_prediction, tf.float32), 1)
# accuracy2 = tf.reduce_mean(all_labels_true)

session = tf.Session()

saver = tf.train.Saver()
if os.path.exists(os.path.join(OUTPUT_DIR, 'two_faces.meta')):
	saver.restore(
		sess=session, 
		save_path=os.path.join(OUTPUT_DIR, 'two_faces'))
else:
	session.run(tf.global_variables_initializer())

train_data = np.load('output/two_faces/inception_resnet_v2/fc_value_list_train.npy')
train_label = np.load('output/two_faces/inception_resnet_v2/labels_train.npy')
test_data = np.load('output/two_faces/inception_resnet_v2/fc_value_list_test.npy')
test_label = np.load('output/two_faces/inception_resnet_v2/labels_test.npy')

def random_batch():
	index = np.random.choice(TRAIN_SIZE, size=BATCH_SIZE, replace=False)
	x_batch = train_data[index]
	y_batch = train_label[index]
	return x_batch, y_batch

def optimize(num_iterations):
	for i in range(num_iterations):
		x_batch, y_batch = random_batch()

		i_global, _ = session.run(
			[global_step, optimizer], feed_dict={x: x_batch, y_true: y_batch})

		if (i_global % 1000 == 0) or (i_global == num_iterations - 1):
			batch_acc = session.run(accuracy, {x: test_data, y_true: test_label})
			msg = "Global Step: {0:>6}, Accuracy: {1:>4.1%}"
			print(msg.format(i_global, batch_acc))
			# batch_acc, batch_prec, batch_recall = session.run(
			# 	[accuracy, precision, recall], 
			# 	feed_dict={x: x_batch, y_true: y_batch})
			# msg = "Global Step: {0:>6}, Accuracy: {1:>4.1%}, Precision: {2:4.1%}, Recall: {3:4.1%}"
			# print(msg.format(i_global, batch_acc, batch_prec, batch_recall))

			saver.save(sess=session, save_path=os.path.join(OUTPUT_DIR, 'two_faces'))

def print_accuracy():
	test_acc, label_pred = session.run([accuracy, label_correct_prediction], {x: test_data, y_true: test_label})
	print("Accuracy on test-set: {0:.1%}".format(test_acc))
	print(label_pred)
	# test_acc, test_prec, test_recall = session.run(
	# 	[accuracy, precision, recall], 
	# 	feed_dict={x: test_data, y_true: test_label})
	# print("Accuracy on test-set: {0:.1%}".format(test_acc))
	# print("Precision on test-set: {0:.1%}".format(test_prec))
	# print("Recall on test-set: {0:.1%}".format(test_recall))

print_accuracy()
optimize(50000)
print_accuracy()