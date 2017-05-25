import numpy as np
import os
import csv
import sys

from dataset import one_hot_encoded

data_name = "kaggle"

data_path = "../dataset/kaggle/"

output_path = "output/kaggle/"

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
		headers = next(datareader)

		i = 0
		row = next(datareader)
		while row[2] == 'Training':
			msg = "\r- Processing image: {0:>6} / {1}".format(i+1, _num_images_train)

			sys.stdout.write(msg)
			sys.stdout.flush()

			pixels = np.fromstring(row[1], dtype=float, sep=' ')
			image = np.concatenate((pixels, pixels, pixels))
			image = image.reshape(image_size, image_size, num_channels)
			images[i] = image

			cls[i] = row[0]

			row = next(datareader)
			i += 1

	print()

	return images, cls, one_hot_encoded(
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
		headers = next(datareader)

		row = next(datareader)
		while (row[2] == 'Training'):
			row = next(datareader)

		i = 0
		while row[2] == 'PublicTest':
			msg = "\r- Processing image: {0:>6} / {1}".format(i+1, _num_images_test)

			sys.stdout.write(msg)
			sys.stdout.flush()

			pixels = np.fromstring(row[1], dtype=float, sep=' ')
			image = np.concatenate((pixels, pixels, pixels))
			image = image.reshape(image_size, image_size, num_channels)
			images[i] = image

			cls[i] = row[0]

			row = next(datareader)
			i += 1

	print()

	return images, cls, one_hot_encoded(
		class_numbers=cls,
		num_classes=num_classes)