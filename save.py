from __future__ import print_function
import tensorflow as tf
import argparse
import os
import random
import imageio
import sys
import numpy as np
import shutil
import Evaluation_script_
from PIL import Image
import matplotlib.image as mpimg

#from tensorflow.examples.tutorials.mnist import input_data

def readimage(folder, index):
    path = os.path.join(folder, str(index) + '.png')
    print(path)
    image = imageio.imread(path)
    #image = image[:,:,1]   # extract the first layer pixel of image
    print("shape == ", image.shape)    # 128*128
    return image

def readmask(folder, index):
    path = os.path.join(folder, str(index) + '.png')
    print(path)
    image[:,:,0] = imageio.imread(path)
    image[:,:,1] = imageio.imread(path)
    image[:,:,2] = imageio.imread(path)
    #image = image[:,:]   # extract the first layer pixel of image
    print("shape == ", image.shape)    # 128*128
    return image

def readnormal(folder, index):
    path = os.path.join(folder, str(index) + '.png')
    print(path)
    image = imageio.imread(path)
    image = image[:,:,:]   # extract the first layer pixel of image
    print("shape == ", image.shape)    # 128*128
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


def evaluation1(prediction, mask, normal):

    mean_angle_error = 0
    total_pixels = 0
    pic_num = prediction.get_shape()[0]
    print("pic_num == ", pic_num)
    show = prediction
    pre = tf.multiply(tf.ones_like(prediction), 0.5)
    nor = tf.multiply(tf.ones_like(normal), 0.5)
    prediction = tf.multiply(tf.subtract(tf.divide(prediction, 255.0), pre), 2)
    normal = tf.multiply(tf.subtract(tf.divide(normal, 255.0), nor), 2)

    for pn in range(0,pic_num):

        total_pixels += tf.count_nonzero(mask[pn,:,:,0])

        a11 = tf.reduce_sum(prediction[pn,:,:,:] * prediction[pn,:,:,:], 2)
        a22 = tf.reduce_sum(normal[pn,:,:,:] * normal[pn,:,:,:], 2)
        a12 = tf.reduce_sum(prediction[pn,:,:,:] * normal[pn,:,:,:], 2)

        cos_dist = a12 / tf.sqrt(a11 * a22)
        cos_dist = tf.where(tf.is_nan(cos_dist),tf.ones_like(cos_dist)*(-1), cos_dist);
        cos_dist = tf.clip_by_value(cos_dist, -1, 1)
        #cos_dist = cos_dist * mask[pn,:,:,0]

        angle_error = tf.acos(cos_dist)
        mean_angle_error += tf.reduce_sum(angle_error)

    print("mean_angle_error == ", mean_angle_error)
    print("total_pixels == ", total_pixels)
    total_pixels = tf.cast(total_pixels, tf.float32)
    return mean_angle_error / total_pixels, show


def evaluation(prediction, mask, normal):
    show = prediction
    pic_num = prediction.get_shape()[0]
    print("pic_num=",pic_num)
    loss = 0
    prediction = prediction / 255.0
    print("scalar")
    normal = normal / 255.0
    nor = normal*mask
    pre = prediction*mask
    for pn in range(0,pic_num):
        loss += tf.norm(normal[pn,:,:,:] - prediction[pn,:,:,:], ord='euclidean')
    return loss, show

global test_num 
global choose_num 
test_num = 100
choose_num = 10

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

'''
color_image = tf.reshape(color, [-1, 128, 128, 1])
mask_image = tf.reshape(color, [-1, 128, 128, 1])
normal_image = tf.reshape(color, [-1, 128, 128, 1])
# print(x_image.shape)  # [n_samples, 28,28,1]
'''

keep_prob = tf.placeholder(tf.float32)


## conv1 layer ##
W_conv0 = weight_variable([3,3,3,64]) # patch 3x3, in size 1, out size 128
b_conv0 = bias_variable([64])
h_conv0 = tf.nn.relu(conv2d(color_image, W_conv0) + b_conv0) # output size 128x128x128 
                                     
## conv2 layer ##
W_conv0 = weight_variable([3,3, 64, 64]) # patch 3x3, in size 128, out size 256
b_conv0 = bias_variable([64])
h_conv0 = tf.nn.relu(conv2d(h_conv0, W_conv0) + b_conv0) # output size 128x128x256

h_conv1 = tf.nn.max_pool(h_conv0, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#h_conv1 = tf.image.resize_images(h_conv1, [h_conv1.get_shape().as_list()[1]*2, h_conv1.get_shape().as_list()[2]*2])

## conv3 layer ##
W_conv1 = weight_variable([3,3, 64, 128]) # patch 3x3, in size 256, out size 256
b_conv1 = bias_variable([128])
h_conv1 = tf.nn.relu(conv2d(h_conv1, W_conv1) + b_conv1) # output size 128x128x256

## conv4 layer ##
W_conv1 = weight_variable([3,3, 128, 128]) # patch 3x3, in size 256, out size 128
b_conv1 = bias_variable([128])
h_conv1 = tf.nn.relu(conv2d(h_conv1, W_conv1) + b_conv1) # output size 128x128x128
  
h_conv2 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#h_conv2 = tf.image.resize_images(h_conv2, [h_conv2.get_shape().as_list()[1]*2, h_conv2.get_shape().as_list()[2]*2])

## conv5 layer ##
W_conv2 = weight_variable([3,3, 128, 256]) # patch 3x3, in size 256, out size 128
b_conv2 = bias_variable([256])
h_conv2 = tf.nn.relu(conv2d(h_conv2, W_conv2) + b_conv2) # output size 128x128x128
 
## conv6 layer ##
W_conv2 = weight_variable([3,3, 256, 256]) # patch 3x3, in size 128, out size 3
b_conv2 = bias_variable([256])
h_conv2 = tf.nn.relu(conv2d(h_conv2, W_conv2) + b_conv2) # output size 128x128x3
h_conv3 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#h_conv3 = tf.image.resize_images(h_conv3, [h_conv3.get_shape().as_list()[1]*2, h_conv3.get_shape().as_list()[2]*2])

## conv7 layer ##
W_conv3 = weight_variable([3,3, 256, 512]) # patch 3x3, in size 256, out size 128
b_conv3 = bias_variable([512])
h_conv3 = tf.nn.relu(conv2d(h_conv3, W_conv3) + b_conv3) # output size 128x128x128

## conv8 layer ##
W_conv3 = weight_variable([3,3, 512, 512]) # patch 3x3, in size 128, out size 3
b_conv3 = bias_variable([512])
h_conv3 = tf.nn.relu(conv2d(h_conv3, W_conv3) + b_conv3) # output size 128x128x3
h_conv4 = tf.nn.max_pool(h_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

## conv9 layer ##
W_conv4 = weight_variable([3,3, 512, 1024]) # patch 3x3, in size 256, out size 128
b_conv4 = bias_variable([1024])
h_conv4 = tf.nn.relu(conv2d(h_conv4, W_conv4) + b_conv4) # output size 128x128x128

## conv10 layer ##
W_conv4 = weight_variable([3,3, 1024, 512]) # patch 3x3, in size 128, out size 3
b_conv4 = bias_variable([512])
h_conv4 = tf.nn.relu(conv2d(h_conv4, W_conv4) + b_conv4) # output size 128x128x3
print("h_conv4'shape = ", h_conv4.get_shape())

uph_conv0 = tf.image.resize_images(h_conv4, [h_conv4.get_shape().as_list()[1]*2, h_conv4.get_shape().as_list()[2]*2])
uph_conv0 = tf.concat([uph_conv0, h_conv3],3)
print("uph_conv0'shape = ", uph_conv0.get_shape())

upW_conv0 = weight_variable([3,3, 1024, 512]) # patch 3x3, in size 256, out size 128
upb_conv0 = bias_variable([512])
uph_conv0 = tf.nn.relu(conv2d(uph_conv0, upW_conv0) + upb_conv0) # output size 128x128x128
 
## conv5 layer ##
upW_conv0 = weight_variable([3,3, 512, 256]) # patch 3x3, in size 128, out size 3
upb_conv0 = bias_variable([256])
uph_conv0 = tf.nn.relu(conv2d(uph_conv0, upW_conv0) + upb_conv0) # output size 128x128x3
uph_conv1 = tf.image.resize_images(uph_conv0, [uph_conv0.get_shape().as_list()[1]*2, uph_conv0.get_shape().as_list()[2]*2])

uph_conv1 = tf.concat([uph_conv1, h_conv2],3)

## conv11 layer ##
upW_conv1 = weight_variable([3,3, 512, 256]) # patch 3x3, in size 256, out size 128
upb_conv1 = bias_variable([256])
uph_conv1 = tf.nn.relu(conv2d(uph_conv1, upW_conv1) + upb_conv1) # output size 128x128x128

## conv5 layer ##
upW_conv1 = weight_variable([3,3, 256, 128]) # patch 3x3, in size 128, out size 3
upb_conv1 = bias_variable([128])
uph_conv1 = tf.nn.relu(conv2d(uph_conv1, upW_conv1) + upb_conv1) # output size 128x128x3
uph_conv2 = tf.image.resize_images(uph_conv1, [uph_conv1.get_shape().as_list()[1]*2, uph_conv1.get_shape().as_list()[2]*2])

uph_conv2 = tf.concat([uph_conv2, h_conv1],3)

upW_conv2 = weight_variable([3,3, 256, 128]) # patch 3x3, in size 256, out size 128
upb_conv2 = bias_variable([128])
uph_conv2 = tf.nn.relu(conv2d(uph_conv2, upW_conv2) + upb_conv2) # output size 128x128x128

upW_conv2 = weight_variable([3,3, 128, 64]) # patch 3x3, in size 256, out size 128
upb_conv2 = bias_variable([64])
uph_conv2 = tf.nn.relu(conv2d(uph_conv2, upW_conv2) + upb_conv2) # output size 128x128x128

uph_conv3 = tf.image.resize_images(uph_conv2, [uph_conv2.get_shape().as_list()[1]*2, uph_conv2.get_shape().as_list()[2]*2])
uph_conv3 = tf.concat([uph_conv3, h_conv0],3)

upW_conv3 = weight_variable([3,3, 128, 64]) # patch 3x3, in size 256, out size 128
upb_conv3 = bias_variable([64])
uph_conv3 = tf.nn.relu(conv2d(uph_conv3, upW_conv3) + upb_conv3) # output size 128x128x128

upW_conv3 = weight_variable([3,3, 64, 64]) # patch 3x3, in size 256, out size 128
upb_conv3 = bias_variable([64])
uph_conv3 = tf.nn.relu(conv2d(uph_conv3, upW_conv3) + upb_conv3) # output size 128x128x128

upW_conv4 = weight_variable([3,3, 64, 3]) # patch 3x3, in size 256, out size 128
upb_conv4 = bias_variable([3])
prediction = tf.nn.relu(conv2d(uph_conv3, upW_conv4) + upb_conv4) # output size 128x128x128


 
# the error between prediction and real data

cross_entropy,show = evaluation(prediction, mask_image, normal_image)
print(cross_entropy)
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
saver = tf.train.Saver()



sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from

init = tf.global_variables_initializer()
#init = tf.initialize_all_variables()
sess.run(init)
save_path = saver.save(sess,"train/model／save_net.ckpt")
print("Save to path:", save_path)
for i in range(5):

    batch_index = random.sample(range(0, test_num), choose_num)    # create random number to select the batch randomly
    print("batch_index == ", batch_index)
    batch_xs = image_all[batch_index, :, :, :]

    mask_xs = mask_all[batch_index, :, :, :]

    print("batch_xs.shape == ", batch_xs.shape)
    batch_ys = normal_all[batch_index, :, :, :]
    print("batch_ys.shape == ", batch_ys.shape)
    _, cross, pred = sess.run([train_step, cross_entropy, show], feed_dict={color_image: batch_xs, mask_image: mask_xs, normal_image: batch_ys, keep_prob: 0.5})
    saver.save(sess = sess, save_path = save_path)
    print("cross_entropy = ", cross)
    #print("prediction ==", pred)


##create the folder to save the prediction images
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



