from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

import tensorflow as tf

import dataset_utils

_NUM_SHARDS = 7

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

def _get_filenames(file_db):
	filenames = []
	file = open(file_db, 'r')
	for line in file:
		row = line.split('.')
		filenames.append(row[0])
	return filenames

def _get_dataset_filename(dataset_dir, split_name, shard_id):
	output_filename = 'aflw_%s_%05d-of-%05d.tfrecord' % (
		split_name, shard_id, _NUM_SHARDS)
	return os.path.join(dataset_dir, output_filename)

def _convert_dataset(split_name, filenames, dataset_dir):
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

						for i in range(2):
							image_data = tf.gfile.FastGFile(
								dataset_dir + '/data/' + filenames[i] + '_' + str(i + 1) + '.png', 'r').read()
							height, width = image_reader.read_image_dims(sess, image_data)

							example = dataset_utils.image_to_tfexample(
								image_data, 'png', height, width, -1)
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

def run():
	dataset_dir = '../dataset/two_faces'

	if _dataset_exists(dataset_dir):
		print('Dataset files already exist. Exiting without re-creating them.')
		return

	training_filenames = _get_filenames(os.path.join(dataset_dir, 'label/training.txt'))
	testing_filenames = _get_filenames(os.path.join(dataset_dir, 'label/testing.txt'))

	_convert_dataset('train', training_filenames, dataset_dir)
	_convert_dataset('test', testing_filenames, dataset_dir)

run()