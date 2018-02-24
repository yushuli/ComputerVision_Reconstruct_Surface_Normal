from __future__ import print_function
import tensorflow as tf
import argparse
import os
import random
import imageio
import sys
import numpy as np
import shutil
#import PIL.Image
#from io import BytesIO
#from IPython.display import Image, display
import Evaluation_script_
from PIL import Image
import matplotlib.image as mpimg

#from tensorflow.examples.tutorials.mnist import input_data

def readimage(folder, index):
	path = os.path.join(folder, str(index) + '.png')
	print(path)
	image = imageio.imread(path)
	#image = image[:,:,1]   # extract the first layer pixel of image
	#print("shape == ", image.shape)    # 128*128
	return image

def readmask(folder, index):
	path = os.path.join(folder, str(index) + '.png')
	print(path)
	image[:,:,0] = imageio.imread(path)
	image[:,:,1] = imageio.imread(path)
	image[:,:,2] = imageio.imread(path)
	#image = image[:,:]   # extract the first layer pixel of image
	#print("shape == ", image.shape)    # 128*128
	return image

def readnormal(folder, index):
	path = os.path.join(folder, str(index) + '.png')
	print(path)
	image = imageio.imread(path)
	image = image[:,:,:]   # extract the first layer pixel of image
	#print("shape == ", image.shape)    # 128*128
	return image

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	#The generated values follow a normal distribution with specified mean and standard deviation
	
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)

	return tf.Variable(initial)

def conv2d(x, W):
	# stride [1, x_movement, y_movement, 1]
	# Must have strides[0] = strides[3] = 1

	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')   # will nor shrink the size

def evaluation(prediction, mask, normal):
	pic_num = prediction.get_shape()[0]
	print("pic_num=",pic_num)
	loss = 0
	prediction = prediction / 255.0
	print("scalar")
	normal = normal / 255.0
	nor = normal*mask
	pre = prediction*mask
	for pn in range(0,pic_num):
		loss += tf.norm(normal[pn,:,:,:] - prediction[pn,:,:,:])
	loss = loss / 200
	return loss

global test_num 
global choose_num 
test_num = 5000
choose_num = 200

image_all = np.zeros((test_num, 128, 128, 3),dtype = float)
mask_all = np.zeros((test_num, 128, 128, 3),dtype = float)
normal_all = np.zeros((test_num, 128, 128, 3),dtype = float)

folder_image = './train/color'
for index in range(0,test_num):
	image = readimage(folder_image, index)
	#print(image.get_shape())
	image_all[index, :, :, :] = image 
print("All images have been read!")

folder_mask = './train/mask'
for index in range(0,test_num):
	image = readmask(folder_mask, index)
	mask_all[index, :, :, :] = image
print("All masks have been read!")

folder_normal = './train/normal'
for index in range(0,test_num):
	image = readnormal(folder_normal, index)  
	normal_all[index, :, :, :] = image
print("ALl surface normals have been read!")


# define placeholder for inputs to network
color_image = tf.placeholder(tf.float32, [choose_num, 128, 128, 3])    # 128x128x3
mask_image = tf.placeholder(tf.float32, [choose_num, 128, 128, 3])     # 128x128x3
normal_image = tf.placeholder(tf.float32, [choose_num, 128, 128, 3])   # 128x128x3


keep_prob = tf.placeholder(tf.float32)


## conv1 layer ##
W_conv1 = weight_variable([3,3,3,32]) # patch 3x3, in size 1, out size 128
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(color_image, W_conv1) + b_conv1) # output size 128x128x128 
print("conv1 == ", h_conv1.get_shape())                                      

## conv2 layer ##
W_conv2 = weight_variable([3,3, 32, 128]) # patch 3x3, in size 128, out size 256
b_conv2 = bias_variable([128])
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2) # output size 128x128x256
print("conv2 == ", h_conv2.get_shape())  

## conv3 layer ##
W_conv3 = weight_variable([3,3, 128, 256]) # patch 3x3, in size 256, out size 256
b_conv3 = bias_variable([256])
h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3) # output size 128x128x256
print("conv3 == ", h_conv3.get_shape())  

## conv4 layer ##
W_conv4 = weight_variable([3,3, 256, 256]) # patch 3x3, in size 256, out size 128
b_conv4 = bias_variable([256])
h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4) # output size 128x128x128
print("conv4 == ", h_conv4.get_shape())  

## conv5 layer ##
W_conv5 = weight_variable([3,3, 256, 128]) # patch 3x3, in size 256, out size 128
b_conv5 = bias_variable([128])
h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5) # output size 128x128x128
print("conv5 == ", h_conv5.get_shape())  

#conv6 layer ##
W_conv6 = weight_variable([3,3, 128, 3]) # patch 3x3, in size 256, out size 128
b_conv6 = bias_variable([3])
prediction = tf.nn.tanh(conv2d(h_conv5, W_conv6) + b_conv6) # output size 128x128x128
print("conv6 == ", prediction.get_shape()) 
# the error between prediction and real data

cross_entropy = evaluation(prediction, mask_image, normal_image)
print(cross_entropy)
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)


sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from

init = tf.global_variables_initializer()
#init = tf.initialize_all_variables()
sess.run(init)

for i in range(500):

	batch_index = random.sample(range(0, test_num), choose_num)    # create random number to select the batch randomly
	print("batch_index == ", batch_index)
	batch_xs = image_all[batch_index, :, :, :]

	mask_xs = mask_all[batch_index, :, :, :]

	print("batch_xs.shape == ", batch_xs.shape)
	batch_ys = normal_all[batch_index, :, :, :]
	print("batch_ys.shape == ", batch_ys.shape)
	_, cross, pred = sess.run([train_step, cross_entropy, prediction], feed_dict={color_image: batch_xs, mask_image: mask_xs, normal_image: batch_ys, keep_prob: 0.5})
	print("cross_entropy = ", cross)
	print("prediction.shape ==", prediction.shape)





selected_dir = ['train/prediction_selected', 'train/mask_selected', 'train/normal_selected']

for dir in selected_dir:
    if os.path.exists(dir):
        shutil.rmtree(dir)

    os.makedirs(dir)


#Save the images
for index in range(choose_num):
    result = Image.fromarray((pred[index, :, :, :]).astype(np.uint8))
    #print("result.shape ==", result.get_shape())
    result.save('train/prediction_selected/' + str(batch_index[index]) + '.png')

    mask_selected = Image.open('train/mask/' + str(batch_index[index]) + '.png')
    mask_selected.save('train/mask_selected/' + str(batch_index[index]) + '.png')

    mask_selected = Image.open('train/normal/' + str(batch_index[index]) + '.png')
    mask_selected.save('train/normal_selected/' + str(batch_index[index]) + '.png')


#Compute MAE

prediction_folder = './train/prediction_selected'
normal_folder = './train/normal_selected'
mask_folder = './train/mask_selected'
mae = Evaluation_script_.evaluate(prediction_folder, normal_folder, mask_folder,)
print("Final MAE ==", mae)
