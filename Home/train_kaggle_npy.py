import numpy as np
import tensorflow as tf
import os
import cv2
import glob

import vgg16
import config

sample_paths = glob.glob(config.KAGGLE_DATA_DIR + '*.jpg')
sample_paths.sort()

count = 0;

with tf.Session() as sess:
	image_list = tf.placeholder("float", [1, 224, 224, 3])
	vgg = vgg16.Vgg16(config.VGG16_NPY_PATH)
	with tf.name_scope("content_vgg"):
		vgg.build(image_list)

	for path in sample_paths:
		image = cv2.imread(path)
		if image != None:
			image = cv2.resize(image, (224, 224))
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			image = image.reshape(1, 224, 224, 3)

			feat = sess.run(vgg.relu6, feed_dict={image_list: image})

			dest_file = path.replace(config.CELEBFACE_DATA_DIR, config.OUT_DIR + 'kaggle/')
			parent_fd = os.path.dirname(dest_file)
			if not os.path.exists(parent_fd):
				os.makedirs(parent_fd)

			np.save(dest_file, feat)

			count += 1
			if count % 10 == 0:
				print(count, '\t', path)