import tensorflow as tf
import os
import cv2
import numpy as np
import prettytensor as pt

import inception
import aflw
import celeb
import kaggle

data_path = '../dataset/img/'

output_path = 'output/2_faces/'

inception.maybe_download()
model = inception.Inception()

def get_two_faces(row):
	file_name = row[0]
	
	if int(row[1]) > 0:
		face_1_x = int(row[1])
	else:
		face_1_x = 0
	if int(row[2]) > 0:
		face_1_y = int(row[2])
	else:
		face_1_y = 0
	face_1_w = int(row[3])
	face_1_h = int(row[4])

	if int(row[5]) > 0:
		face_2_x = int(row[5])
	else:
		face_2_x = 0
	if int(row[6]) > 0:
		face_2_y = int(row[6])
	else:
		face_2_y = 0
	face_2_w = int(row[7])
	face_2_h = int(row[8])

	if os.path.isfile(data_path + row[0]):
		image = cv2.imread(data_path + row[0])
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		face_1 = image[face_1_y: face_1_y + face_1_h, face_1_x: face_1_x + face_1_w]
		face_2 = image[face_2_y: face_2_y + face_2_h, face_2_x: face_2_x + face_2_w]

		# resize faces
		def resize_face(face):
			image_h, image_w, dump = face.shape
			if (image_h > 100 or image_w > 100):
				ratio = 0.0
				dim = None
				if image_h > image_w:
					ratio = float(100) / image_h
					dim = (int(image_w * ratio), 100)
				else:
					ratio = float(100) / image_w
					dim = (100, int(image_h * ratio))

				face = cv2.resize(face, dim, interpolation=cv2.INTER_AREA)
			return face

		face_1 = resize_face(face_1)
		face_2 = resize_face(face_2)

	return face_1, face_2

def get_relation_traits(row):
	return np.array(object=row[9: 17], dtype=float)

def get_spatial_cues(row):
	file_name = row[0]
	
	face_1_x = int(row[1])
	face_1_y = int(row[2])
	face_1_w = int(row[3])
	face_1_h = int(row[4])

	face_2_x = int(row[5])
	face_2_y = int(row[6])
	face_2_w = int(row[7])
	face_2_h = int(row[8])

	return np.array(object=[
		(face_1_x - face_2_x) / face_1_w, 
		(face_1_y - face_2_y) / face_2_w,
		face_1_w / face_2_w], 
		dtype=float)

def get_softmax_layer(transfer_values, dataset):
	tf.reset_default_graph()

	# Placeholder Variables
	transfer_len = model.transfer_len
	x = tf.placeholder(tf.float32, shape=[None, transfer_len], name='x')
	y_true = tf.placeholder(tf.float32, shape=[None, dataset.num_classes], name='y_true')
	y_true_cls = tf.argmax(y_true, dimension=1)

	# Neural Network
	x_pretty = pt.wrap(x)	# Wrap the transfer-values as a Pretty Tensor object.
	with pt.defaults_scope(activation_fn=tf.nn.relu):
		y_pred, loss = x_pretty.fully_connected(size=1024, name='layer_fc1').softmax_classifier(num_classes=dataset.num_classes, labels=y_true)

	# Optimization Method
	global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
	optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step)

	# Classification Accuracy
	y_pred_cls = tf.argmax(y_pred, dimension=1)
	# correct_prediction = tf.equal(y_pred_cls, y_true_cls)
	correct_prediction = tf.diag_part(tf.matmul(y_pred, tf.transpose(y_true)))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	prob_list = None
	with tf.Session() as session:
		saver = tf.train.Saver()
		saver.restore(sess=session, save_path=(dataset.output_path + dataset.data_name))

		prob_list = y_pred.eval({x: transfer_values}, session)
		session.close()

	return prob_list

file = open(data_path + 'Label/testing.txt', 'r')

spatial_cue_list = np.zeros(shape=[700, 3], dtype=float)
relation_traits_list = np.zeros(shape=[700, 8], dtype=float)
# face_1_list = [None] * 700
# face_2_list = [None] * 700

face_1_list = np.full(shape=[700, 100,100, 3], fill_value=255, dtype=float)
face_2_list = np.full(shape=[700, 100,100, 3], fill_value=255, dtype=float)

for i in range(700):
	str = file.readline()
	row = str.split(" ")

	spatial_cue_list[i] = get_spatial_cues(row)
	print(i)

	face_1, face_2 = get_two_faces(row)
	face_1_list[i, 0: face_1.shape[0], 0: face_1.shape[1]] = face_1[0: face_1.shape[0], 0: face_1.shape[1]]
	face_2_list[i, 0: face_2.shape[0], 0: face_2.shape[1]] = face_2[0: face_2.shape[0], 0: face_2.shape[1]]
	
	relation_traits_list[i] = get_relation_traits(row)

file.close()

if not os.path.exists(output_path):
	os.makedirs(output_path)

from inception import transfer_values_cache
transfer_values_face_1 = transfer_values_cache(cache_path=output_path + 'inception_face_1_test.pkl', images=face_1_list, model=model)
transfer_values_face_2 = transfer_values_cache(cache_path=output_path + 'inception_face_2_test.pkl', images=face_2_list, model=model)

conc = np.concatenate(
	(
		spatial_cue_list,
		get_softmax_layer(transfer_values_face_1, aflw), 
		get_softmax_layer(transfer_values_face_1, kaggle), 
		get_softmax_layer(transfer_values_face_1, celeb),
		get_softmax_layer(transfer_values_face_2, aflw), 
		get_softmax_layer(transfer_values_face_2, kaggle), 
		get_softmax_layer(transfer_values_face_2, celeb)), 
	axis=1)

np.save(output_path + 'softmax_vector_test.npy', conc)
np.save(output_path + 'relation_traits_vector_test.npy', relation_traits_list)
