import os
import math

import cv2

import numpy as np

import tensorflow as tf

from sklearn.model_selection import train_test_split

datafiles = os.listdir("imgs")

train_files, valid_files = train_test_split(datafiles, train_size=0.8)

def load_img(filename):
    img = cv2.imread(filename)
    img = img[:300,:300,:3]
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    #Remove aim (network was taking conclusion based on it, I am color blind so I did not realize the aim changes color)
    aim_thickness = 1
    aim_length = 16
    #debug_img[150-offset:150+offset, 150-offset:150+offset, :] = 0
    img[150-aim_length:150+aim_length, 150-aim_thickness:150+aim_thickness, :] = 0
    img[150-aim_thickness:150+aim_thickness, 150-aim_length:150+aim_length, :] = 0
    
    return img, int(filename[5] == 'p')

def get_image_batches(files_list, batch_size):
    for i in range(0, len(files_list), batch_size):
        x_batch, y_batch = list(zip(*[load_img("imgs/" + filename) for filename in files_list[i:i+batch_size]]))
        x_batch = np.array(x_batch)
        yield x_batch, y_batch
        
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

def create_optimizer(logits, labels, learning_rate=0.001):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    loss = tf.reduce_mean(loss)
    
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    return optimizer, loss

def create_perform_metrics(outputs, labels):
    
    outputs = tf.cast(outputs > 0.5, tf.bool)
    labels = tf.cast(labels, tf.bool)
    
    acc, acc_update = tf.metrics.accuracy(outputs, labels)
    recall, recall_update = tf.metrics.recall(outputs, labels)
    
    return acc_update, recall_update

def reset_metrics_variables():
    return [tf.assign(v,0) for v in tf.local_variables() if 'accuracy' in v.name or 'recall' in v.name]

notpressed_ratio = len([f for f in valid_files if 'not' in f]) / len(valid_files)
print(notpressed_ratio)

N_EPOCHS = 2

BATCH_SIZE = 128

n_batches = math.ceil(len(train_files) / BATCH_SIZE)

#-------------------------------------------------#

tf.reset_default_graph()

inputs, targets = create_inputs()

logits, outputs = create_conv_net(inputs)
optimizer, loss = create_optimizer(logits, targets, 0.001)

#valid_logits, valid_outputs = create_conv_net(valid_dataset_x_batch, reuse=True)
metrics_update = create_perform_metrics(outputs, targets)
reset_metrics = reset_metrics_variables()

saver = tf.train.Saver()

print("Initializing session...")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    for e in range(N_EPOCHS):
        #train_files, valid_files
        
        i_batch = 0
        
        for x_batch, y_batch in get_image_batches(train_files, BATCH_SIZE):
            
            i_batch += 1
    
            _, loss_value = sess.run([optimizer, loss], feed_dict={
                inputs: x_batch,
                targets: y_batch
            })
        
            print("Epoch {}/{} \t Batch {}/{} \t Loss: {}".format(e+1, N_EPOCHS, i_batch, n_batches, loss_value))
            
            
        print("Calculating performance...")
        sess.run(reset_metrics)
        for valid_x_batch, valid_y_batch in get_image_batches(valid_files, BATCH_SIZE):
    
            acc_loss_value = sess.run(loss, feed_dict={
                inputs: valid_x_batch,
                targets: valid_y_batch
            })
    
            acc_value, recall_value = sess.run(metrics_update, feed_dict={
                inputs: valid_x_batch,
                targets: valid_y_batch
            })
        
        print("Accuracy: {} \t Recall: {} \t Loss: {}".format(acc_value, recall_value, acc_loss_value))
        
        saver.save(sess, "checkpoints/model.ckpt")