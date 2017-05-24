import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import os
import cv2
import csv

import inception
import config

def load_training_data():
	images = np.zeros(shape=[28709, 48, 48, 3], dtype=float)
	cls = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
	labels = np.zeros(shape=[28709, 7], dtype=float)

	with open(config.DATA_DIR + 'kaggle/fer2013.csv', 'rt') as csvfile:
		datareader = csv.reader(csvfile, delimiter =',')
		headers = next(datareader)

		for i in range(28709):
			row = next(datareader)

			if row[2] == 'Training':
				image = cv2.imread(config.DATA_DIR + 'kaggle/Training/' + str(i + 1) + '.jpg')
				images[i] = image
				labels[i, int(row[0])] = 1.0

				if i % 100 == 0:
					print(i)

	return images, cls, labels

def load_test_data():
	images = np.zeros(shape=[3589, 48, 48, 3], dtype=float)
	labels = np.zeros(shape=[3589, 7], dtype=float)

	with open(config.DATA_DIR + 'kaggle/fer2013.csv', 'rt') as csvfile:
		datareader = csv.reader(csvfile, delimiter =',')
		headers = next(datareader)

		row = next(datareader)
		while (row[2] != 'PublicTest'):
			row = next(datareader)

		for i in range(3589):
			image = cv2.imread(config.DATA_DIR + 'kaggle/PublicTest/' + str(i + 28710) + '.jpg')
			images[i] = image
			print(int(row[0]))
			labels[i, int(row[0])] = 1.0

			row = next(datareader)

			# if i % 100 == 0:
			# 	print(i)

	return images, labels

def save_labels_train(labels_train):
	np.save('transfer_values/kaggle', labels_train)

if __name__ == '__main__':
	images_train, cls_train, labels_train = load_training_data()
	images_test, labels_test = load_test_data()
	inception.maybe_download()
	# print(images_train)
	# print(labels_train)

	model = inception.Inception()

	from inception import transfer_values_cache
	file_path_cache_train = os.path.join(config.TRANSFER_VALUES_DIR, 'inception_kaggle_train.pkl')

	print("Processing Inception transfer-values for training-images ...")
	transfer_values_train = transfer_values_cache(cache_path=file_path_cache_train, images=images_train, model=model)

	#############################
	file_path_cache_test = os.path.join(config.TRANSFER_VALUES_DIR, 'inception_kaggle_test.pkl')

	print("Processing Inception transfer-values for testing-images ...")
	transfer_values_test = transfer_values_cache(cache_path=file_path_cache_test, images=images_test, model=model)

	save_labels_train(labels_train)
	np.save('transfer_values/test_kaggle', labels_test)