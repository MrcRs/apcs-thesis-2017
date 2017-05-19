# coding: utf-8

'''
This script creates 3-channel gray images from FER 2013 dataset.
It has been done so that the CNNs designed for RGB images can 
be used without modifying the input shape. 

This script requires two command line parameters:
1. The path to the CSV file
2. The output directory

It generates the images and saves them in three directories inside 
the output directory - Training, PublicTest, and PrivateTest. 
These are the three original splits in the dataset. 
'''


import os
import csv
import argparse
import numpy as np 
import cv2

w, h = 48, 48
image = np.zeros((h, w), dtype=np.uint8)
id = 1
with open('fer2013.csv', 'rt') as csvfile:
    datareader = csv.reader(csvfile, delimiter =',')
    headers = next(datareader)
    print (headers) 
    for row in datareader:  
        emotion = row[0]
        pixels = np.fromstring(row[1], dtype=np.int, sep=' ')
        usage = row[2]
        #print emotion, type(pixels[0]), usage

        image = pixels.reshape(w, h)
        #print image.shape

        stacked_image = np.dstack((image,) * 3)
        #print stacked_image.shape

        image_folder = os.path.join('./', usage)
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        image_file =  os.path.join(image_folder , str(id) + '.jpg')
        # scipy.misc.imsave(image_file, stacked_image)
        cv2.imwrite(image_file, stacked_image)
        id += 1 
        if id % 100 == 0:
            print('Processed {} images'.format(id))

print("Finished processing {} images".format(id))
