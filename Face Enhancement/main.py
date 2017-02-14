import dataset
import facenet.inception_resnet_v1 as facenet

import tensorflow as tf
import os
import shutil
import datetime

layers = tf.contrib.layers
losses = tf.contrib.losses
arg_scope = tf.contrib.framework.arg_scope

num_layer = 18
num_channels = 64
batch_size = 32
weight_decay = 1e-8

learning_rate = 0.1
lr_decay_step = 1000
lr_decay_rate = 0.95
theta = 0.01

log_directory = '/tmp/log'

if os.path.isdir(log_directory):
        shutil.rmtree(log_directory)

set_input = dataset.FileStreamer('x', '/media/jaehwan/5F454EF74C1ECD04/selfie-dataset/image_c/*')
set_input = dataset.ImageStreamer(set_input,
                                  shape=dataset.Coordinate(120, 120, 3),
                                  color_space='lab')
set_base = dataset.FileStreamer('x_base', '/media/jaehwan/5F454EF74C1ECD04/selfie-dataset/image_c_btf/*')
set_base = dataset.ImageStreamer(set_base,
                                 shape=dataset.Coordinate(120, 120, 1),
                                 color_space='gray')
set_target = dataset.FileStreamer('y', '/media/jaehwan/5F454EF74C1ECD04/selfie-dataset/image_c_dn/*')
set_target = dataset.ImageStreamer(set_target,
                                   shape=dataset.Coordinate(120, 120, 3),
                                   color_space='lab')
set = dataset.JoinStreamer([set_input, set_base, set_target])
set = dataset.ParallelDataset(set, batch_size)
x, x_base, y = set.ops

x /= 100
y /= 100
x = tf.expand_dims(x[..., 0], dim=3)
y = tf.expand_dims(y[..., 0], dim=3)
x_detail = x - x_base
y_detail = y - x_base


# Generator network
with arg_scope([layers.conv2d],
               weights_regularizer=layers.l2_regularizer(weight_decay),
               biases_regularizer=layers.l2_regularizer(weight_decay),
               variables_collections=tf.GraphKeys.TRAINABLE_VARIABLES,
               outputs_collections=tf.GraphKeys.ACTIVATIONS):
    net = tf.concat(3, [x_base, x_detail])
    net = layers.conv2d(inputs=net,
                        num_outputs=num_channels,
                        kernel_size=[3, 3],
                        scope='conv0')
    for i in range(1, num_layer):
        net_temp = layers.convolution2d(inputs=net,
                                        num_outputs=num_channels,
                                        kernel_size=[3, 3],
                                        stride=1,
                                        scope='conv'+str(i))
        net = tf.add(net, net_temp, name='conv'+str(i)+'/add')
    net = layers.conv2d(net, 1, [3, 3],
                        scope='conv'+str(num_layer),
                        activation_fn=None)
    y_detail_ = net
    y_disp_ = tf.clip_by_value(net + x_base, 0, 1)

loss_op = losses.mean_squared_error(y_detail_, y_detail)
# Facenet network
# y_semantic_, _ = facenet.inference(images=y_,
#                                    keep_probability=1.0,
#                                    phase_train=False)
# y_semantic, __ = facenet.inference(images=y,
#                                    keep_probability=1.0,
#                                    phase_train=False,
#                                    reuse=True)

# image_loss_op = losses.mean_squared_error(y_, y)
# semantic_loss_op = losses.mean_squared_error(y_semantic_, y_semantic)
# loss_op = image_loss_op
# loss_op = image_loss_op + semantic_loss_op

tf.image_summary('x', x)
tf.image_summary('y', y)
tf.image_summary('y_', y_disp_)
tf.scalar_summary('loss', loss_op)
# tf.scalar_summary('image_loss', image_loss_op)
# tf.scalar_summary('semantic_loss', semantic_loss_op)
for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
    tf.histogram_summary(var.name, var)
for act in tf.get_collection(tf.GraphKeys.ACTIVATIONS):
    tf.histogram_summary(act.name, act)
tf.histogram_summary('x_detail', x_detail)
tf.histogram_summary('y_detail', y_detail)

with tf.name_scope('optimizer'):
    global_step_op = tf.Variable(0, trainable=False)
    learning_rate_op = tf.train.exponential_decay(learning_rate=learning_rate,
                                                  global_step=global_step_op,
                                                  decay_steps=lr_decay_step,
                                                  decay_rate=lr_decay_rate,
                                                  staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss_op,
                                                 tf.trainable_variables())
    threshold = theta / learning_rate
    capped_grads_and_vars = []
    for grad, var in grads_and_vars:
        gv = (tf.clip_by_value(grad, -threshold, threshold), var)
        capped_grads_and_vars.append(gv)
    train_op = optimizer.apply_gradients(capped_grads_and_vars,
                                         global_step=global_step_op)

tf.scalar_summary('learning_rate', learning_rate_op)

summary_op = tf.merge_all_summaries()
saver = tf.train.Saver()

init_op = tf.initialize_all_variables()
sess = tf.Session()

# Load facenet model
# facenet_var_list = []
# for var in tf.get_collection(tf.GraphKeys.VARIABLES):
#     if var.name.startswith(u'InceptionResnetV1'):
#         facenet_var_list.append(var)
# saver_facenet = tf.train.Saver(var_list=facenet_var_list)
# saver_facenet.restore(sess, './facenet/model-20161116-234200.ckpt-80000')

writer = tf.train.SummaryWriter(log_directory)
writer.add_graph(sess.graph)

print('Start train')
try:
    sess.run(init_op)
    set.start(sess)
    for _ in range(1000000):
        global_step = tf.train.global_step(sess, global_step_op)
        sess.run(train_op)
        if global_step % 10 == 0:
            print('[Iter {}] {}'.format(global_step, datetime.datetime.now()))
        if global_step >= 300 and global_step % 100 == 0:
            summary_str = sess.run(summary_op)
            writer.add_summary(summary_str, global_step)
        if global_step >= 10000 and global_step % 10000 == 0:
            saver.save(sess,
                       os.path.join(log_directory,
                                    'model_{}.ckpt'.format(str(global_step))))
    print('Train finished')
except Exception as e:
    set.terminate()
    sess.close()
    raise e

sess.close()
