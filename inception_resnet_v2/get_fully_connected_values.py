import tensorflow as tf
import os
import cv2
import numpy as np

from nets import nets_factory
from preprocessing import preprocessing_factory

AFLW_TRAIN_MODEL_PATH = 'output/aflw-models/inception_resnet_v2/all/model.ckpt-10000'
KAGGLE_TRAIN_MODEL_PATH = 'output/kaggle-models/inception_resnet_v2/all/model.ckpt-15000'

DATA_DIR = '../dataset/two_faces'
TRAIN_DIR = 'output/two_faces/inception_resnet_v2'

TRAIN_SIZE = 7226
TEST_SIZE = 790

def get_two_faces(row):
	file_name = row[0]
	
	face1_x1 = int(row[1])
	face1_y1 = int(row[2])
	face1_x2 = face1_x1 + int(row[3])
	face1_y2 = face1_y1 + int(row[4])
	if face1_x1 < 0:
		face1_x1 = 0
	if face1_y1 < 0:
		face1_y1 = 0

	face2_x1 = int(row[5])
	face2_y1 = int(row[6])
	face2_x2 = face2_x1 + int(row[7])
	face2_y2 = face2_y1 + int(row[8])
	if face2_x1 < 0:
		face2_x1 = 0
	if face2_y1 < 0:
		face2_y1 = 0

	if os.path.isfile(DATA_DIR + '/img/' + row[0]):
		print(row[0])
		image = cv2.imread(DATA_DIR + '/img/' + row[0])
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		face_1 = image[face1_y1: face1_y2, face1_x1: face1_x2]
		face_2 = image[face2_y1: face2_y2, face2_x1: face2_x2]

		face_1 = cv2.resize(face_1, (299, 299), interpolation=cv2.INTER_AREA)
		face_2 = cv2.resize(face_2, (299, 299), interpolation=cv2.INTER_AREA)

		faces = np.concatenate((face_1, face_2))
		faces = faces.reshape(2, 299, 299, 3)
		faces = 2 * faces / 255.0 - 1
	return faces

def get_relation_traits(row):
	return np.array(object=row[9: 17], dtype=int)

def get_relation_traits_list(num_classes, db_path, size):
	relation_trait_list = np.zeros(shape=[size, num_classes], dtype=int)

	file = open(db_path)
	i = 0
	for line in file:
		row = line.split(' ')
		relation_trait_list[i] = get_relation_traits(row)
		i += 1

	return relation_trait_list

def get_fc_value_list(num_classes, model_path, db_path, size):
	tf.reset_default_graph()

	fc_value_list = np.empty(shape=[size, 1536 * 2])

	x = tf.placeholder(tf.float32, shape=[None, 299, 299, 3], name='x')
	network_fn = nets_factory.get_network_fn(
		'inception_resnet_v2',
		num_classes=num_classes,
		is_training=False)
	logits, end_points = network_fn(x)

	with tf.Session() as session:
		saver = tf.train.Saver()
		saver.restore(sess=session, save_path=model_path)

		file = open(db_path)
		i = 0
		for line in file:
			row = line.split(' ')
			faces = get_two_faces(row)
			fc_value = session.run(end_points['PreLogitsFlatten'], feed_dict={x: faces})
			fc_value = fc_value.reshape(1536 * 2)
			fc_value_list[i] = fc_value
			i += 1

	print(fc_value_list)
	print(fc_value_list.shape)
	return fc_value_list

def save_train_npy():
	aflw_fc_value_list = get_fc_value_list(
		3, 
		AFLW_TRAIN_MODEL_PATH, 
		os.path.join(DATA_DIR, 'label/training.txt'), 
		TRAIN_SIZE)

	kaggle_fc_value_list = get_fc_value_list(
		7, 
		KAGGLE_TRAIN_MODEL_PATH, 
		os.path.join(DATA_DIR, 'label/training.txt'), 
		TRAIN_SIZE)

	fc = np.concatenate((aflw_fc_value_list, kaggle_fc_value_list), axis=1)
	np.save(os.path.join(TRAIN_DIR, 'fc_value_list_train.npy'), fc)

	# np.save(os.path.join(TRAIN_DIR, 'fc_value_list_train.npy'), aflw_fc_value_list)

	relation_trait_list = get_relation_traits_list(
		8, os.path.join(DATA_DIR, 'label/training.txt'),TRAIN_SIZE)
	np.save(os.path.join(TRAIN_DIR, 'labels_train.npy'), relation_trait_list)

def save_test_npy():
	aflw_fc_value_list = get_fc_value_list(
		3, 
		AFLW_TRAIN_MODEL_PATH, 
		os.path.join(DATA_DIR, 'label/testing.txt'), 
		TEST_SIZE)

	kaggle_fc_value_list = get_fc_value_list(
		7, 
		KAGGLE_TRAIN_MODEL_PATH, 
		os.path.join(DATA_DIR, 'label/testing.txt'), 
		TEST_SIZE)

	fc = np.concatenate((aflw_fc_value_list, kaggle_fc_value_list), axis=1)
	np.save(os.path.join(TRAIN_DIR, 'fc_value_list_test.npy'), fc)

	# np.save(os.path.join(TRAIN_DIR, 'fc_value_list_test.npy'), aflw_fc_value_list)

	relation_trait_list = get_relation_traits_list(
		8, os.path.join(DATA_DIR, 'label/testing.txt'),TEST_SIZE)
	np.save(os.path.join(TRAIN_DIR, 'labels_test.npy'), relation_trait_list)

save_train_npy()
save_test_npy()
# def get_two_fc_values(faces, num_classes, save_path):
# 	x = tf.placeholder(tf.float32, shape=[None, 299, 299, 3], name='x')

# 	network_fn = nets_factory.get_network_fn(
# 		'inception_resnet_v2',
# 		num_classes=num_classes,
# 		is_training=False)
# 	logits, _ = network_fn(x)

# 	with tf.Session() as session:
# 		saver = tf.train.Saver()
# 		saver.restore(sess=session, save_path=save_path)
# 		fc_value = session.run(logits, feed_dict={x: faces})
# 		fc_value = fc_value.reshape(num_classes * 2)

# 	return fc_value

# def get_fc_value_list(db_name):
# 	fc_value_list = np.empty(shape=[TRAIN_SIZE, 20])

# 	file = open(os.path.join(DATA_DIR, db_name))

# 	i = 0
# 	for line in file:
# 		row = line.split(' ')
# 		faces = get_two_faces(row)
# 		aflw_fc_value = get_two_fc_values(
# 			faces, 3, os.path.join(AFLW_TRAIN_DIR, 'model.ckpt-35000'))
# 		kaggle_fc_value = get_two_fc_values(
# 			faces, 7, os.path.join(KAGGLE_TRAIN_DIR, 'model.ckpt-60000'))

# 		fc_value = np.concatenate((aflw_fc_value, kaggle_fc_value), axis=0)
# 		fc_value_list[i] = fc_value

# 	print(fc_value_list)
# 	print(fc_value_list.shape)
# 	np.save(os.path.join(TRAIN_DIR, 'fc_value_list.npy'), fc_value_list)

# get_fc_value_list('label/training.txt')
