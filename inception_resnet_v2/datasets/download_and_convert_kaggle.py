from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import csv
import numpy as np
from six.moves import urllib
import tensorflow as tf

from datasets import dataset_utils

_DATA_URL = ''
_CSV_FILENAME = 'fer2013.csv'
_TRAIN_IMAGES_FILENAME = 'train_images.npy'
_TRAIN_LABELS_FILENAME = 'train_labels.npy'
_TEST_IMAGES_FILENAME = 'test_images.npy'
_TEST_LABELS_FILENAME = 'test_labels.npy'

_IMAGE_SIZE = 48
_NUM_CHANNELS = 3

_CLASS_NAMES = [
	'angry',
	'disgust',
	'fear',
	'happy',
	'sad',
	'surprise',
	'neutral'
]

def _get_images(filename, num_images):
	print('Getting images from: ', filename)
	images = np.load(filename)
	return images

def _get_labels(filename, num_labels):
	print('Getting labels from: ', filename)
	labels = np.load(filename)
	return labels

def _add_to_tfrecord(data_filename, labels_filename, num_images, tfrecord_writer):
	images = _get_images(data_filename, num_images)
	labels = _get_labels(labels_filename, num_images)

	shape = (_IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS)
	with tf.Graph().as_default():
		image = tf.placeholder(dtype=tf.uint8, shape=shape)
		encoded_png = tf.image.encode_png(image)

		with tf.Session('') as sess:
			for j in range(num_images):
				sys.stdout.write('\r>> Converting image %d/%d' % (j + 1, num_images))
				sys.stdout.flush()

				png_string = sess.run(encoded_png, feed_dict={image: images[j]})
				example = dataset_utils.image_to_tfexample(
					png_string, 'png'.encode(), _IMAGE_SIZE, _IMAGE_SIZE, labels[j])
				tfrecord_writer.write(example.SerializeToString())

def _get_output_filename(dataset_dir, split_name):
	return '%s/kaggle_%s.tfrecord' % (dataset_dir, split_name)

def _csv_to_npy(dataset_dir):
	train_images = np.zeros(shape=[28709, _IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS], dtype=np.uint8)
	train_labels = np.zeros(shape=[28709], dtype=np.uint8)
	test_images = np.zeros(shape=[7178, _IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS], dtype=np.uint8)
	test_labels = np.zeros(shape=[7178], dtype=np.uint8)

	with open(dataset_dir + '/' + _CSV_FILENAME) as file:
		reader = csv.reader(file, delimiter=',')

		train_count = 0
		test_count = 0
		for row in reader:
			if row[2] != 'Usage':
				pixels = np.fromstring(row[1], dtype=np.uint8, sep=' ')
				pixels = pixels.reshape(_IMAGE_SIZE, _IMAGE_SIZE)
				image = np.dstack((pixels,) * 3)
				if row[2] == 'Training':
					train_images[train_count] = image
					train_labels[train_count] = row[0]
					train_count += 1
				else:
					test_images[test_count] = image
					test_labels[test_count] = row[0]
					test_count += 1

	np.save(os.path.join(dataset_dir, _TRAIN_IMAGES_FILENAME), train_images)
	np.save(dataset_dir + '/' + _TRAIN_LABELS_FILENAME, train_labels)
	np.save(dataset_dir + '/' + _TEST_IMAGES_FILENAME, test_images)
	np.save(dataset_dir + '/' + _TEST_LABELS_FILENAME, test_labels)

def _clean_up_temporary_files(dataset_dir):
	for filename in [_TRAIN_IMAGES_FILENAME, _TRAIN_LABELS_FILENAME, _TEST_IMAGES_FILENAME, _TEST_LABELS_FILENAME]:
		filepath = os.path.join(dataset_dir, filename)
		tf.gfile.Remove(filepath)

def run(dataset_dir):
	if not tf.gfile.Exists(dataset_dir):
		tf.gfile.MakeDirs(dataset_dir)

	training_filename = _get_output_filename(dataset_dir, 'train')
	testing_filename = _get_output_filename(dataset_dir, 'test')

	if tf.gfile.Exists(training_filename) and tf.gfile.Exists(testing_filename):
		print('Dataset files already exist. Exiting without re-creating them.')
		return

	_csv_to_npy(dataset_dir)

	with tf.python_io.TFRecordWriter(training_filename) as tfrecord_writer:
		data_filename = os.path.join(dataset_dir, _TRAIN_IMAGES_FILENAME)
		labels_filename = os.path.join(dataset_dir, _TRAIN_LABELS_FILENAME)
		_add_to_tfrecord(data_filename, labels_filename, 28709, tfrecord_writer)

	with tf.python_io.TFRecordWriter(testing_filename) as tfrecord_writer:
		data_filename = os.path.join(dataset_dir, _TEST_IMAGES_FILENAME)
		labels_filename = os.path.join(dataset_dir, _TEST_LABELS_FILENAME)
		_add_to_tfrecord(data_filename, labels_filename, 7178, tfrecord_writer)

	labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
	dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

	_clean_up_temporary_files(dataset_dir)
	print('\nFinish converting the Kaggle dataset!')