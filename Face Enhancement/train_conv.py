import dataset
import facenet.inception_resnet_v1 as facenet

import tensorflow as tf
import os
import shutil
import datetime

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

if os.path.isdir(log_directory):
        shutil.rmtree(log_directory)

set_input = dataset.FileStreamer('x', '/media/jaehwan/5F454EF74C1ECD04/selfie-dataset/image_c_120/*')
set_input = dataset.ImageStreamer(set_input,
                                  shape=dataset.Coordinate(120, 120, 3),
                                  color_space='lab')
set_base = dataset.FileStreamer('x_base', '/media/jaehwan/5F454EF74C1ECD04/selfie-dataset/image_c_btf/*')
set_base = dataset.ImageStreamer(set_base,
                                 shape=dataset.Coordinate(120, 120, 1),
                                 color_space='gray')
set_target = dataset.FileStreamer('y', '/media/jaehwan/5F454EF74C1ECD04/selfie-dataset/image_g/*')
set_target = dataset.ImageStreamer(set_target,
                                   shape=dataset.Coordinate(120, 120, 3),
                                   color_space='lab')
set = dataset.JoinStreamer([set_input, set_base, set_target])
set = dataset.ParallelDataset(set, batch_size)
x_input, x_base, y_input = set.ops
x_input /= 100
y_input /= 100


x_l, x_a, x_b = tf.split(split_dim=3, num_split=3, value=x_input)
y_l, y_a, y_b = tf.split(split_dim=3, num_split=3, value=y_input)
x_detail = x_l - x_base
y_detail = y_l - x_base

# x = tf.concat(concat_dim=3, values=[x_base, x_detail, x_a, x_b])
x = tf.concat(concat_dim=3, values=[x_base, x_detail])
# y = tf.concat(concat_dim=3, values=[y_detail, y_a, y_b])
y = tf.concat(concat_dim=3, values=[y_detail])


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
    y_ = net
    y_disp_ = tf.clip_by_value(net + x_base, 0, 1)

loss_op = losses.mean_squared_error(y_, y)

tf.image_summary('x', x_l)
tf.image_summary('y_', y_disp_)
tf.image_summary('y', y_l)
tf.scalar_summary('loss', loss_op)
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
        if global_step >= 5000 and global_step % 5000 == 0:
            saver.save(sess,
                       os.path.join(log_directory,
                                    'model_{}.ckpt'.format(str(global_step))))
    print('Train finished')
except Exception as e:
    set.terminate()
    sess.close()
    raise e

sess.close()
