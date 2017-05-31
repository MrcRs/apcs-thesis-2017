import numpy as np
import os
import cv2
import sys

data_name = "celebface"

data_path = "../dataset/celebface/"

output_path = "output/celebface/"

file_path_cache_train = 'inception_celebface_train.pkl'

file_path_cache_test = 'inception_celebface_test.pkl'

image_size = 75

num_channels = 3

num_classes = 7

_num_images_train = 15000	# 7202 + 7385 + 6656

_num_images_test = 1500

def load_class_names():
	return ['Smiling', 'Mouth_Slightly_Open', 'Young', 'Goatee', 'No_Beard', 'Sideburns', '5_o_Clock_Shadow']

def load_training_data():
	images = np.zeros(shape=[_num_images_train, image_size, image_size, num_channels], dtype=float)
	cls = np.zeros(shape=[_num_images_train, num_classes], dtype=float)

	file = open(data_path + 'list_attr_celeba.txt')
	file.readline()
	str_labels = file.readline()
	labels = str_labels.split(" ")

	for i in range(_num_images_train):
		str_row = file.readline()
		row = str_row.split()

		if os.path.isfile(data_path + 'img_align_celeba/' + row[0]) and (i < _num_images_train):
			msg = "\r- Processing image: {0:>6} / {1}".format(i+1, _num_images_train)
			sys.stdout.write(msg)
			sys.stdout.flush()

			image, cls[i] = load_image_and_class(row, labels, file_path_cache_train)
			if image is not None:
				image_h, image_w, dump = image.shape
				images[i, 0: image_h, 0: image_w] = image[0: image_h, 0:image_w]

	return images, cls

def load_test_data():
	images = np.zeros(shape=[_num_images_test, image_size, image_size, num_channels], dtype=float)
	cls = np.zeros(shape=[_num_images_test, num_classes], dtype=float)

	file = open(data_path + 'list_attr_celeba.txt')
	file.readline()
	str_labels = file.readline()
	labels = str_labels.split(" ")

	i = 0;
	while (i < _num_images_train):
		file.readline()
		i += 1

	for i in range(_num_images_test):
		str_row = file.readline()
		row = str_row.split(" ")

		if os.path.isfile(data_path + 'img_align_celeba/' + row[0]) and (i < _num_images_test):
			msg = "\r- Processing image: {0:>6} / {1}".format(i+1, _num_images_test)
			sys.stdout.write(msg)
			sys.stdout.flush()

			image, cls[i] = load_image_and_class(row, labels, file_path_cache_test)
			if image is not None:
				image_h, image_w, dump = image.shape
				images[i, 0: image_h, 0: image_w] = image[0: image_h, 0:image_w]

	return images, cls


def create_dir():
	if not os.path.exists(output_path):
		os.makedirs(output_path)

def load_image_and_class(row, labels, pkl_path):
	image = None
	if os.path.exists(output_path + pkl_path) == False:
		image = cv2.imread(data_path + 'img_align_celeba/' + row[0])
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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

	cls = [0.0] * num_classes
	for name in load_class_names():
		index = labels.index(name)
		if row[index] == '1':
			cls[load_class_names().index(name)] = 1.0
	cls = np.array(cls)
	return image, cls