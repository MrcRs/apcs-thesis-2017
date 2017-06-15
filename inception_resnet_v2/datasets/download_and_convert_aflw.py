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

_NUM_VALIDATION = 2438

_RANDOM_SEED = 0

_NUM_SHARDS = 40

_CLASS_NAMES = {
	'right',
	'front',
	'left'
}

class ImageReader(object):
	"""docstring for ImageReader"""
	def __init__(self):
		self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
		self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)
		self._decode_png_data = tf.placeholder(dtype=tf.string)
		self._decode_png = tf.image.decode_png(self._decode_png_data, channels=3)

	def read_image_dims(self, sess, image_data, data):
		print(data[0])
		if _get_image_type(data[0]) == 'jpg':
			image = self.decode_jpeg(sess, image_data, data)
		else:
			image = self.decode_png(sess, image_data, data)
		return image.shape[0], image.shape[1]

	def decode_jpeg(self, sess, image_data, data):
		# crop_data = tf.image.crop_to_bounding_box(
		# 	self._decode_jpeg, data[3], data[2], data[5], data[4])
		# if data[5] > 50 or data[4] > 50:
		# 	ratio = min(50.0 / data[5], 50.0 / data[4])
		# 	crop_data = tf.image.resize_images(
		# 		crop_data, [int(data[5] * ratio), int(data[4] * ratio)])

		# image = sess.run(crop_data, feed_dict={self._decode_jpeg_data: image_data})
		image = sess.run(self._decode_jpeg, feed_dict={self._decode_jpeg_data: image_data})
		assert len(image.shape) == 3
		assert image.shape[2] == 3
		return image

	def decode_png(self, sess, image_data, data):
		# crop_data = tf.image.crop_to_bounding_box(
		# 	self._decode_png, data[3], data[2], data[5], data[4])
		# if data[5] > 50 or data[4] > 50:
		# 	ratio = min(50.0 / data[5], 50.0 / data[4])
		# 	crop_data = tf.image.resize_images(
		# 		crop_data, [int(data[5] * ratio), int(data[4] * ratio)])

		# image = sess.run(crop_data, feed_dict={self._decode_png_data: image_data})
		image = sess.run(self._decode_png, feed_dict={self._decode_png_data: image_data})
		assert len(image.shape) == 3
		assert image.shape[2] == 3
		return image

def _get_data_table(dataset_dir):
	select_string = "faceimages.filepath, facepose.yaw, facerect.x, facerect.y, facerect.w, facerect.h, faceimages.width, faceimages.height"
	from_string = "faceimages, faces, facepose, facerect"
	where_string = "faces.face_id = facepose.face_id and faces.file_id = faceimages.file_id and faces.face_id = facerect.face_id"
	query_string = "SELECT " + select_string + " FROM " + from_string + " WHERE " + where_string

	conn = sqlite3.connect(os.path.join(dataset_dir, 'aflw.sqlite'))
	c = conn.cursor()
	result = c.execute(query_string)

	data = [None] * 24384

	idx = 0
	for row in result:
		tmp = list(row)
		if tmp[2] < 0:
			tmp[2] = 0
		if tmp[3] < 0:
			tmp[3] = 0
		if tmp[4] > tmp[6] - tmp[2]:
			tmp[4] = tmp[6] - tmp[2]
		if tmp[5] > tmp[7] - tmp[3]:
			tmp[5] = tmp[7] - tmp[3]
		data[idx] = tmp
		idx += 1

	return data

def _get_dataset_filename(dataset_dir, split_name, shard_id):
	output_filename = 'aflw_%s_%05d-of-%05d.tfrecord' % (
		split_name, shard_id, _NUM_SHARDS)
	return os.path.join(dataset_dir, output_filename)

def _get_image_type(filename):
	l = len(filename)
	return filename[l - 3:]

def _get_class_id(data):
	if data[1] >= 0.5:
		return 2
	elif data[1] <=-0.5:
		return 0
	else:
		return 1

def _convert_dataset(split_name, data, dataset_dir):
	assert split_name in ['train', 'test']

	num_per_shard = int(math.ceil(len(data) / float(_NUM_SHARDS)))

	with tf.Graph().as_default():
		image_reader = ImageReader()

		with tf.Session('') as sess:
			for shard_id in range(_NUM_SHARDS):
				output_filename = _get_dataset_filename(
					dataset_dir, split_name, shard_id)

				with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
					start_ndx = shard_id * num_per_shard
					end_ndx = min((shard_id+1) * num_per_shard, len(data))
					for i in range(start_ndx, end_ndx):
						sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
							i+1, len(data), shard_id))
						sys.stdout.flush()

						filepath = os.path.join(dataset_dir, data[i][0])
						if tf.gfile.Exists(filepath):						
							image_data = tf.gfile.FastGFile(filepath, 'r').read()
							height, width = image_reader.read_image_dims(sess, image_data, data[i])
							
							class_id = _get_class_id(data[i])

							example = dataset_utils.image_to_tfexample(
								image_data, bytes(_get_image_type([data[i][0]])), height, width, class_id)
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

	data_table = _get_data_table(dataset_dir)
	
	random.seed(_RANDOM_SEED)
	random.shuffle(data_table)
	train_data = data_table[_NUM_VALIDATION:]
	test_data = data_table[:_NUM_VALIDATION]

	_convert_dataset('train', train_data, dataset_dir)
	_convert_dataset('test', test_data, dataset_dir)

	labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
	dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
