from win32api import GetKeyState
import win32api
import time

import win32com.client as comctl
wsh = comctl.Dispatch("WScript.Shell")

from grabscreen import grab_screen
import cv2

import random

import tensorflow as tf

import matplotlib.pyplot as plt


def transform_img(img):
    #img = img[:300,:300,:3]
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    #Remove aim (network was taking conclusion based on it, I am color blind so I did not realize the aim changes color)
    aim_thickness = 1
    aim_length = 16
    #debug_img[150-offset:150+offset, 150-offset:150+offset, :] = 0
    img[150-aim_length:150+aim_length, 150-aim_thickness:150+aim_thickness, :] = 0
    img[150-aim_thickness:150+aim_thickness, 150-aim_length:150+aim_length, :] = 0
    
    return img

def create_inputs():
    inputs = tf.placeholder(shape=[None, 300, 300, 3], dtype=tf.float32)
    targets = tf.placeholder(shape=[None], dtype=tf.float32)
    
    return inputs, targets

def create_conv_net(image_batch, reuse=False):
    with tf.variable_scope("conv_net", reuse=reuse):
        
        image_batch /= 255 #Normalize image data
        
        conv1 = tf.layers.conv2d(image_batch, filters=8, kernel_size=[5,5], strides=[1, 1], padding='SAME',
                                kernel_initializer=None, bias_initializer=tf.zeros_initializer(), activation=tf.nn.relu)
        conv1 = tf.layers.max_pooling2d(conv1, pool_size=[5,5], strides=[2,2], padding='SAME')
        #print(conv1.get_shape())
        
        conv2 = tf.layers.conv2d(conv1, filters=16, kernel_size=[3,3], strides=[1, 1], padding='SAME',
                                kernel_initializer=None, bias_initializer=tf.zeros_initializer(), activation=tf.nn.relu)
        conv2 = tf.layers.max_pooling2d(conv2, pool_size=[3,3], strides=[2,2], padding='SAME')
        #print(conv2.get_shape())
        
        conv3 = tf.layers.conv2d(conv2, filters=32, kernel_size=[3,3], strides=[1, 1], padding='SAME',
                                kernel_initializer=None, bias_initializer=tf.zeros_initializer(), activation=tf.nn.relu)
        conv3 = tf.layers.max_pooling2d(conv3, pool_size=[3,3], strides=[2,2], padding='SAME')
        #print(conv3.get_shape())
        
        conv4 = tf.layers.conv2d(conv3, filters=64, kernel_size=[3,3], strides=[1, 1], padding='SAME',
                                kernel_initializer=None, bias_initializer=tf.zeros_initializer(), activation=tf.nn.relu)
        conv4 = tf.layers.max_pooling2d(conv4, pool_size=[3,3], strides=[2,2], padding='SAME')
        #print(conv4.get_shape())
        
        conv5 = tf.layers.conv2d(conv4, filters=128, kernel_size=[3,3], strides=[1, 1], padding='SAME',
                                kernel_initializer=None, bias_initializer=tf.zeros_initializer(), activation=tf.nn.relu)
        conv5 = tf.layers.max_pooling2d(conv5, pool_size=[3,3], strides=[2,2], padding='SAME')
        #print(conv5.get_shape())

        flatten_layer = tf.layers.flatten(conv5)
        
        h1 = tf.layers.dense(flatten_layer, 5000, tf.nn.relu)
        h2 = tf.layers.dense(h1, 1000, tf.nn.relu)
        h3 = tf.layers.dense(h2, 256, tf.nn.relu)
        
        logits = tf.squeeze(tf.layers.dense(h3, 1), axis=1)
        
        outputs = tf.nn.sigmoid(logits)
        
        return logits, outputs


if __name__ == "__main__":
    tf.reset_default_graph()

    inputs, targets = create_inputs()

    logits, outputs = create_conv_net(inputs)

    #saver = tf.train.Saver()

    #sess = tf.Session()

    #saver.restore(sess, "checkpoints/model.ckpt")


    #time.sleep(5)


    while True:

        delay = 1
            
        #screen = grab_screen(region=(650,300,949,599))#left, top, x2, y2, subtract 1 from x2 and y2 to match 300x300pixels
        screen = grab_screen(region=(653,305,949,599))#left, top, x2, y2, subtract 1 from x2 and y2 to match 300x300pixels
        screen = transform_img(screen)

        print(screen.shape)

        plt.imshow(screen)
        plt.show()
        break
        #pred = sess.run(outputs, feed_dict={ inputs: [screen] })

        #print(pred)

        time.sleep(delay)
    