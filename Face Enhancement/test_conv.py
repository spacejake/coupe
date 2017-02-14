import dataset

import cv2
import tensorflow as tf
import numpy as np
import skimage.io
import skimage.color
import skimage.filters
import os

layers = tf.contrib.layers
losses = tf.contrib.losses
arg_scope = tf.contrib.framework.arg_scope

num_channels = 64
num_layer = 12
batch_size = 48
weight_decay = 1e-8

learning_rate = 0.1
lr_decay_step = 1000
lr_decay_rate = 0.8
theta = 0.01

log_directory = 'log-conv'

set_input = dataset.FileStreamer('x', 'images/input.jpg')
set_input = dataset.ImageStreamer(set_input,
                                  shape=dataset.Coordinate(120, 120, 3),
                                  color_space='lab')
set_base = dataset.FileStreamer('x_base', 'images/base.jpg')
set_base = dataset.ImageStreamer(set_base,
                                 shape=dataset.Coordinate(120, 120, 1),
                                 color_space='gray')

set = dataset.JoinStreamer([set_input, set_base])
x_input = tf.placeholder(dtype=tf.float32, shape=[1, 120, 120, 3])
x_base = tf.placeholder(dtype=tf.float32, shape=[1, 120, 120, 1])


x_l, x_a, x_b = tf.split(split_dim=3, num_split=3, value=x_input / 100)
x_detail = x_l - x_base

x = tf.concat(concat_dim=3, values=[x_base, x_detail])


# Generator network
with arg_scope([layers.conv2d, layers.conv2d_transpose],
               kernel_size=[3, 3],
               weights_regularizer=layers.l2_regularizer(weight_decay),
               biases_regularizer=layers.l2_regularizer(weight_decay),
               variables_collections=tf.GraphKeys.TRAINABLE_VARIABLES,
               outputs_collections=tf.GraphKeys.ACTIVATIONS):
    net = x
    net = layers.conv2d(inputs=net,
                        num_outputs=num_channels,
                        scope='conv0')
    for i in range(1, num_layer):
        net_temp = layers.convolution2d(inputs=net,
                                        num_outputs=num_channels,
                                        stride=1,
                                        scope='conv' + str(i))
        net = tf.add(net, net_temp, name='conv' + str(i) + '/add')
    net = layers.conv2d(net,
                        num_outputs=1,
                        scope='conv' + str(num_layer),
                        activation_fn=None)
    net = net + x_base
    net = tf.clip_by_value(net, 0, 1)
    y_ = net

# loss_op = losses.mean_squared_error(y_, y)

saver = tf.train.Saver()

sess = tf.Session()

writer = tf.train.SummaryWriter(log_directory)
# writer.add_graph(sess.graph)

print('Start test')
try:
    saver.restore(sess, os.path.join(log_directory, 'model_50000.ckpt'))
    for datum in set:
        # x_input_val = datum['x'].data
        # x_base_val = datum['x_base'].data
        # y_input_val = datum['y'].data
        x_input_val = np.expand_dims(datum['x'].data, 0)
        x_base_val = np.expand_dims(np.expand_dims(datum['x_base'].data, 0), 3)
        # y_input_val = np.expand_dims(datum['y'].data, 0)

        y_color_val = np.squeeze(x_input_val[..., 1:])
        # y_color_val = np.squeeze(y_input_val[..., 1:])
        # y_guidance = skimage.io.imread('guidance.png')
        # y_guidance = skimage.color.rgb2lab(y_guidance)
        # y_color_val = y_guidance[..., 1:]
        y_color_val[..., 0] *= 1.25
        y_color_val[..., 1] *= 0.85

        # y_val_ = x_base_val + (x_detail_val * 0.8 + y_detail_val * 0.2)
        # y_val_ = y_val_ * 0.2 + cv2.blur(y_val_, (3, 3)) * 0.8
        # y_val_ *= 100
        # y_val_ = np.concatenate((np.expand_dims(y_val_, axis=2),
        #                          np.squeeze(y_input_val[..., 1:])), axis=2)

        y_val_ = sess.run(y_, feed_dict={x_input: x_input_val,
                                         x_base: x_base_val})
        print(y_val_.shape)
        y_val_ = np.squeeze(y_val_)
        y_val_ *= 100
        # y_val_ = 0.85 * y_val_ + 0.15 * np.squeeze(y_input_val[..., 0])
        y_val_ = np.expand_dims(y_val_, axis=2)
        y_val_ = np.concatenate((y_val_, y_color_val),
                                axis=2)
        y_val_ = skimage.color.lab2rgb(y_val_)
        skimage.io.imshow(y_val_)
        skimage.io.show()

        # skimage.io.imsave('input.png', skimage.color.lab2rgb(np.squeeze(x_input_val)))
        # skimage.io.imsave('target.png', skimage.color.lab2rgb(np.squeeze(y_input_val)))
        skimage.io.imsave('output.png', y_val_)

    print('Test finished')
except Exception as e:
    # sess.close()
    raise e

# sess.close()
