import numpy as np
import tensorflow as tf
import os
import glob
import cv2

import vgg16
import config

sample_paths = glob.glob(config.DATA_DIR + '/aia/aia_anger/*.png')
sample_paths.sort()
f = open('samples_paths', 'w')
f.writelines(sample_paths)
f.close()

flog = open('log.txt', 'w')
count = 0

with tf.Session() as sess:
	images = tf.placeholder("float", [1, 224, 224, 3])
	vgg = vgg16.Vgg16(config.VGG16_NPY_PATH)
	with tf.name_scope("content_vgg"):
		vgg.build(images)

	for path in sample_paths:
		img = cv2.imread(path)
		img = cv2.resize(img, (224, 224))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = img.reshape(1, 224, 224, 3)

		feat = sess.run(vgg.relu6, feed_dict={images: img}) # extract feats from ReLU-6 layer.

		dest_file = path.replace(config.DATA_DIR, config.OUT_DIR)
		parent_fd = os.path.dirname(dest_file)
		if not os.path.exists(parent_fd):
			os.makedirs(parent_fd)
		np.save(dest_file, feat)

		count += 1
		flog.write('%06d\t%s\n'%(count, path))
		print (count, '\t', path)
		# print ('%06d\t%s', %(count, path))
