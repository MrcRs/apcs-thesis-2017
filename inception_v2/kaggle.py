import numpy as np
import os
import csv
import sys

from dataset import one_hot_encoded

data_name = "kaggle"

data_path = "../dataset/kaggle/"

output_path = "output/kaggle/"

file_path_cache_train = 'inception_kaggle_train.pkl'

file_path_cache_test = 'inception_kaggle_test.pkl'

image_size = 48

num_channels = 3

num_classes = 7

_num_images_train = 28709

_num_images_test = 3589

def load_class_names():
	class_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
	return class_names

def load_training_data():
	images = np.zeros(
		shape=[_num_images_train, image_size, image_size, num_channels], 
		dtype=float)

	cls = np.zeros(
		shape=[_num_images_train],
		dtype=int)

	with open(data_path + 'fer2013.csv', 'rt') as csvfile:
		datareader = csv.reader(csvfile, delimiter =',')

		i = 0
		for row in datareader:
			if row[2] == 'Training':
				msg = "\r- Processing image: {0:>6} / {1}".format(i+1, _num_images_train)

				sys.stdout.write(msg)
				sys.stdout.flush()

				images[i], cls[i] = load_images_and_cls(row, file_path_cache_train)

				i += 1
			if i == _num_images_train:
				break

	print()

	return images, one_hot_encoded(
		class_numbers=cls, 
		num_classes=num_classes)

def load_test_data():
	images = np.zeros(
		shape=[_num_images_test, image_size, image_size, num_channels], 
		dtype=float)
	cls = np.zeros(
		shape=[_num_images_test],
		dtype=int)

	with open(data_path + 'fer2013.csv', 'rt') as csvfile:
		datareader = csv.reader(csvfile, delimiter =',')

		i = 0
		for row in datareader:
			if row[2] == 'PrivateTest':
				msg = "\r- Processing image: {0:>6} / {1}".format(i+1, _num_images_test)
				sys.stdout.write(msg)
				sys.stdout.flush()

				images[i], cls[i] = load_images_and_cls(row, file_path_cache_test)

				i += 1
			if i == _num_images_train:
				break

	print()

	return images, one_hot_encoded(
		class_numbers=cls,
		num_classes=num_classes)

def create_dir():
	if not os.path.exists(output_path):
		os.makedirs(output_path)

def load_images_and_cls(row, pkl_path):
	image = None
	if os.path.exists(output_path + pkl_path) == False:
		pixels = np.fromstring(row[1], dtype=float, sep=' ')
		image = np.concatenate((pixels, pixels, pixels))
		image = image.reshape(image_size, image_size, num_channels)

	return image, row[0]
