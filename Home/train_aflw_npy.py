import numpy as np
import tensorflow as tf
import os
import cv2
import sqlite3

import vgg16
import config

images_path = config.AFLW_DATA_DIR + 'data/'
storing_path = config.OUT_DIR + 'aflw/'
sql_path = config.AFLW_DATA_DIR + 'aflw.sqlite'

count = 0

conn = sqlite3.connect(sql_path)
c = conn.cursor()

select_string = "faceimages.filepath, faces.face_id, facepose.roll, facepose.pitch, facepose.yaw, facerect.x, facerect.y, facerect.w, facerect.h"
from_string = "faceimages, faces, facepose, facerect"
where_string = "faces.face_id = facepose.face_id and faces.file_id = faceimages.file_id and faces.face_id = facerect.face_id"
query_string = "SELECT " + select_string + " FROM " + from_string + " WHERE " + where_string

with tf.Session() as sess:
	image_list = tf.placeholder("float", [1, 224, 224, 3])
	vgg = vgg16.Vgg16(config.VGG16_NPY_PATH)

	with tf.name_scope("content_vgg"):
		vgg.build(image_list)

	for row in c.execute(query_string):
		input_path = images_path + str(row[0])
		output_path = storing_path + str(row[0])
		print (input_path)

		if os.path.isfile(input_path) == True:
			image = cv2.imread(input_path)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

			image_h, image_w, image_color = image.shape

			roll = row[2]
			pitch = row[3]
			yaw = row [4]

			face_x = row[5]
			face_y = row[6]
			face_w = row[7]
			face_h = row[8]

			if face_x < 0:
				face_x = 0
			if face_y < 0:
				face_y = 0
			if face_w > image_w:
				face_w = image_w
				face_h = image_w
			if face_h > image_h:
				face_h = image_h
				face_w = image_h

			cropped_image = np.copy(image[face_y:face_y+face_h, face_x:face_x+face_w])
			cropped_image = cv2.resize(cropped_image, (224, 224))
			cropped_image = cropped_image.reshape(1, 224, 224, 3)
			print (cropped_image)

			feat = sess.run(vgg.relu6, feed_dict={image_list: cropped_image})

			dest_file = input_path.replace(config.AFLW_DATA_DIR, storing_path)
			parent_fd = os.path.dirname(dest_file)
			if not os.path.exists(parent_fd):
				os.makedirs(parent_fd)

			np.save(dest_file, feat)

			count += 1
			if count % 10 == 0:
				print(count, '\t', input_path)