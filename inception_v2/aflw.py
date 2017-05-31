import numpy as np
import os
import sqlite3
import cv2
import sys

from dataset import one_hot_encoded

data_name = "aflw"

data_path = "../dataset/aflw/"

output_path = "output/aflw/"

file_path_cache_train = 'inception_aflw_train.pkl'

file_path_cache_test = 'inception_aflw_test.pkl'

image_size = 100

num_channels = 3

num_classes = 3

_num_images_train = 15000	# 7202 + 7385 + 6656

_num_images_test = 1500

select_string = "faceimages.filepath, facepose.yaw, facerect.x, facerect.y, facerect.w, facerect.h"
from_string = "faceimages, faces, facepose, facerect"
where_string = "faces.face_id = facepose.face_id and faces.file_id = faceimages.file_id and faces.face_id = facerect.face_id"
query_string = "SELECT " + select_string + " FROM " + from_string + " WHERE " + where_string

def load_class_names():
	return ['left', 'frontal', 'right']

def load_training_data():
	images = np.zeros(shape=[_num_images_train, image_size, image_size, num_channels], dtype=float)
	cls = np.zeros(shape=[_num_images_train], dtype=int)

	conn = sqlite3.connect(data_path + 'aflw.sqlite')
	c = conn.cursor()

	result = c.execute(query_string)

	i = 0
	for row in result:
		if os.path.isfile(data_path + str(row[0])) and (i < _num_images_train):
			msg = "\r- Processing image: {0:>6} / {1}".format(i+1, _num_images_train)
			sys.stdout.write(msg)
			sys.stdout.flush()

			image, cls[i] = load_images_and_cls(row, file_path_cache_train)
			if image is not None:
				image_h, image_w, dump = image.shape
				images[i, 0: image_h, 0: image_w] = image[0: image_h, 0:image_w]

			i += 1

	print()

	return images, one_hot_encoded(class_numbers=cls, num_classes=num_classes)

def load_test_data():
	images = np.zeros(shape=[_num_images_test, image_size, image_size, num_channels], dtype=float)
	cls = np.zeros(shape=[_num_images_test], dtype=int)

	conn = sqlite3.connect(data_path + 'aflw.sqlite')
	c = conn.cursor()

	result = c.execute(query_string)

	i = 0
	for row in result:
		if i < _num_images_train:
			i += 1

		elif os.path.isfile(data_path + str(row[0])) and (i - _num_images_train < _num_images_test):
			msg = "\r- Processing image: {0:>6} / {1}".format(i - _num_images_train + 1, _num_images_test)
			sys.stdout.write(msg)
			sys.stdout.flush()

			image, cls[i - _num_images_train] = load_images_and_cls(row, file_path_cache_test)
			if image is not None:
				image_h, image_w, dump = image.shape
				images[i - _num_images_train, 0: image_h, 0: image_w] = image[0: image_h, 0:image_w]

			i += 1

	print()

	return images, one_hot_encoded(class_numbers=cls, num_classes=num_classes)
	
def create_dir():
	if not os.path.exists(output_path):
		os.makedirs(output_path)

def load_images_and_cls(row, pkl_path):
	image = None
	if os.path.exists(output_path + pkl_path) == False:
		image = cv2.imread(data_path + str(row[0]))
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		image_h, image_w, dump = image.shape

		#Face rectangle coords
		face_x = row[2]
		face_y = row[3]
		face_w = row[4]
		face_h = row[5]

		#Error correction
		if(face_x < 0): face_x = 0
		if(face_y < 0): face_y = 0
		if(face_w > image_w): 
			face_w = image_w
			face_h = image_w
		if(face_h > image_h): 
			face_h = image_h
			face_w = image_h

		image = image[face_y: face_y + face_h, face_x: face_x + face_w]
		image_h, image_w, dump = image.shape
		if (image_h > image_size or image_w > image_size):
			ratio = 0.0
			dim = None
			if image_h > image_w:
				ratio = float(image_size) / image_h
				dim = (int(image_w * ratio), image_size)
			else:
				ratio = float(image_size) / image_w
				dim = (image_size, int(image_h * ratio))

			image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

	if row[1] >= 0.5:
		return image, 2
	elif row[1] <= -0/5:
		return image, 0
	else:
		return image, 1
