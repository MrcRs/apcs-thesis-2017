from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import sqlite3
import array

import tensorflow as tf

from datasets import dataset_utils

_NUM_VALIDATION = 1476

_RANDOM_SEED = 0

_NUM_SHARDS = 13

_CLASS_NAMES = {
	'right',
	'front',
	'left'
}

class ImageReader(object):
	"""docstring for ImageReader"""
	def __init__(self):
		self._decode_png_data = tf.placeholder(dtype=tf.string)
		self._decode_png = tf.image.decode_png(self._decode_png_data, channels=3)

	def read_image_dims(self, sess, image_data):
		image = self.decode_png(sess, image_data)
		return image.shape[0], image.shape[1]

	def decode_png(self, sess, image_data):
		image = sess.run(self._decode_png, feed_dict={self._decode_png_data: image_data})
		assert len(image.shape) == 3
		assert image.shape[2] == 3
		return image

def _get_filenames_and_classes(dataset_dir):
	aflw_root = dataset_dir
	directories = []
	class_names = []
	for filename in os.listdir(aflw_root):
		path = os.path.join(aflw_root, filename)
		if os.path.isdir(path):
			directories.append(path)
			class_names.append(filename)

	photo_filenames = []
	for directory in directories:
		for filename in os.listdir(directory):
			path = os.path.join(directory, filename)
			photo_filenames.append(path)

	return photo_filenames, sorted(class_names)

def _get_dataset_filename(dataset_dir, split_name, shard_id):
	output_filename = 'aflw_%s_%05d-of-%05d.tfrecord' % (
		split_name, shard_id, _NUM_SHARDS)
	return os.path.join(dataset_dir, output_filename)

def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir):
	assert split_name in ['train', 'test']
	num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

	with tf.Graph().as_default():
		image_reader = ImageReader()

		with tf.Session('') as sess:
			for shard_id in range(_NUM_SHARDS):
				output_filename = _get_dataset_filename(
					dataset_dir, split_name, shard_id)

				with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
					start_ndx = shard_id * num_per_shard
					end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
					
					for i in range(start_ndx, end_ndx):
						sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
							i+1, len(filenames), shard_id))
						sys.stdout.flush()

						image_data = tf.gfile.FastGFile(filenames[i], 'r').read()
						height, width = image_reader.read_image_dims(sess, image_data)

						class_name = os.path.basename(os.path.dirname(filenames[i]))
						class_id = class_names_to_ids[class_name]

						example = dataset_utils.image_to_tfexample(
							image_data, 'png', height, width, class_id)
						tfrecord_writer.write(example.SerializeToString())
	sys.stdout.write('\n')
	sys.stdout.flush()

def _dataset_exists(dataset_dir):
	for split_name in ['train', 'test']:
		for shard_id in range(_NUM_SHARDS):
			output_filename = _get_dataset_filename(
				dataset_dir, split_name, shard_id)
			if not tf.gfile.Exists(output_filename):
				return False
	return True

def run(dataset_dir):
	if not tf.gfile.Exists(dataset_dir):
		tf.gfile.MakeDirs(dataset_dir)

	if _dataset_exists(dataset_dir):
		print('Dataset files already exist. Exiting without re-creating them.')
		return

	# maybe download dataset here !!!

	photo_filenames, class_names = _get_filenames_and_classes(dataset_dir)
	class_names_to_ids = dict(zip(class_names, range(len(class_names))))

	random.seed(_RANDOM_SEED)
	random.shuffle(photo_filenames)
	training_filenames = photo_filenames[_NUM_VALIDATION:]
	testing_filenames = photo_filenames[:_NUM_VALIDATION - 1]

	_convert_dataset(
		'train', training_filenames, class_names_to_ids, dataset_dir)
	_convert_dataset(
		'test', testing_filenames, class_names_to_ids, dataset_dir)

	labels_to_class_names = dict(zip(range(len(class_names)), class_names))
	dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

	print('\nFinished converting the AFLW dataset!')