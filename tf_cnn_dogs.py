# ############################################################
# Tensorflow for machine learning 
# CNN for dogs data set
# ############################################################

# image resizing (should apply padding)
# image alpha only reading now jpgs (need to have option for pngs)
# train in rgb instead of grey scale
# 

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import itertools
import glob
import time
#get_ipython().magic(u'matplotlib inline')

#X, b = sess.run(read_images(file_path_regex="./output/testing-images/*.tfrecords", batch_size=5))




file_path_regex="./output/testing-images/*.tfrecords"

#file_path_regex="./output/training-images/*.tfrecords"

def read_images(file_path_regex, batch_size=7):

    files_matched = tf.train.match_filenames_once(file_path_regex)
    filename_queue = tf.train.string_input_producer(files_matched)
    
    reader = tf.TFRecordReader()
    _, value = reader.read(filename_queue)
    
    # features (X) dictionary
    features = tf.parse_single_example(
        value,
        features={
            'label':tf.FixedLenFeature([], tf.string),
            'image':tf.FixedLenFeature([], tf.string)
        })
    
    # definition of shape and data types
    image = tf.reshape(tf.decode_raw(features['image'], tf.uint8), [250,151,1]) # the data was reshaped in write function ... must put the same shape
    label = tf.cast(features['label'], tf.string)
    
    batch_size=7
    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size + 10
    
    image_batch, label_batch = tf.train.shuffle_batch([image, label], 
                                                      batch_size=batch_size, 
                                                      capacity=capacity, 
                                                      min_after_dequeue=min_after_dequeue)
    
    float_image_batch = tf.image.convert_image_dtype(image_batch, tf.float32)
    
    labels = map(lambda x: x.split("/")[-1], glob.glob('./imagenet-dogs/Images/*'))
    # this will translate lables 
    m_ = tf.map_fn(lambda l: tf.cast(tf.where(tf.equal(labels,l)), dtype=tf.float32), label_batch, dtype=tf.float32)
    m_ = tf.cast(tf.reshape(m_, shape=[1,batch_size])[0], dtype=tf.int32)
    
    return float_image_batch, m_
    
#    return float_image_batch, batch_size

# create labels
# --------------------------------------------
# Model
def inference(X):

    # conv1 > pool1 > conv2 > pool2 > full_con1 > full_con2

    # first convolution layer 
    conv_2d_layer_one = tf.contrib.layers.convolution2d(
        inputs=X, 
        num_outputs=32,
        kernel_size=(5,5),
        activation_fn=tf.nn.relu,
        weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32),
        stride=(2,2),
        trainable=True    
        )
    
    pool_layer_one = tf.nn.max_pool(conv_2d_layer_one, ksize = [1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    # second convolution layer
    conv_2d_layer_two = tf.contrib.layers.convolution2d(
       inputs = pool_layer_one, 
       num_outputs=64, 
       kernel_size=(5,5), 
       activation_fn=tf.nn.relu, 
       weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32),
       stride=(1,1),
       trainable=True    
       )    
    
    pool_layer_two = tf.nn.max_pool(value=conv_2d_layer_two , ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    # flatten output vector of size [ batch size and ncol] = ncol nxm from pool layer 2
    flattend_layer_two = tf.reshape(pool_layer_two, [batch_size, -1])    
    
    hidden_layer_three = tf.contrib.layers.fully_connected(inputs=flattend_layer_two, 
                                                         num_outputs=512,
                                                         activation_fn=tf.nn.relu,
                                                         weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32)#tf.random_normal([38912, 512], stddev=0.1)
                                                         )
    #hidden_layer_three.get_shape()
    
    hidden_layer_three_d = tf.nn.dropout(hidden_layer_three, 0.4)
    
    finall_fully_connected = tf.contrib.layers.fully_connected(inputs=hidden_layer_three_d, 
                                                              num_outputs=120, 
                                                              weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32) # tf.truncated_normal([512, 120], stddev=0.1)
                                                              )
    
    return finall_fully_connected


# y =  should be one hot encoded array (measuring distance betrween two vectors)
yhat = inference(X=float_image_batch)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(yhat, m_))
optim = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)


def accuracy():
    
    y_ = np.array(sess.run(tf.arg_max((tf.nn.softmax(yhat)), dimension=1)))
    l_ = np.array(sess.run(m_))
    test_ = y_ == l_
    output = sum(test_) / float(y_.shape[0])
        
    return output, y_, l_

# --------------------------------------------
# Train Model
#saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.initialize_all_variables())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)

for i in range(100):
    _, l = sess.run([optim, loss])
    if i % 10 == 0:
        print l, a
    if i % 20 == 0:
        accuracy()
        



np.argmax(sess.run(tf.nn.softmax(yhat)), axis=1)

sess.run(tf.nn.softmax(yhat))

loss_out = []
ptime = time.time()
for i in range(100000):
    _, l = sess.run([optim, loss])
    #print l    
    if i % 100 == 0:
        loss_out.append(l) 
        print l
    if i % 1000 == 0:
        print i
        saver.save(sess, save_path = "./tf_saved_session/tf_test_cnn_save.ckpt", global_step=i)
elapsed = ptime - time.time()
saver.save(sess, save_path = "./tf_saved_session/tf_test_cnn_save.ckpt", global_step=i)
#sess.close()

plt.plot(loss_out)



np.argmax([0,0,0,1])