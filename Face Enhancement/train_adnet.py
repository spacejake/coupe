import dataset

import tensorflow as tf
import os
import shutil
import datetime
import nets

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

log_directory = '/tmp/log2'

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
x_input, x_base, y_input = set.ops

x_input /= 100
y_input /= 100
x = tf.expand_dims(x_input[..., 0], dim=3)
y = tf.expand_dims(y_input[..., 0], dim=3)
x_detail = x - x_base
y_detail = y - x_base
x = tf.concat(3, [x_base, x_detail, x_input[..., 1:]])
y = tf.concat(3, [y_detail, y_input[..., 1:]])


# Generator network
y_ = nets.generator(x, num_layer, num_channels, weight_decay)
y_disp_ = tf.clip_by_value(y_ + x_base, 0, 1)

batch_norm_params = {
    # Decay for the moving averages.
    'decay': 0.995,
    # epsilon to prevent 0s in variance.
    'epsilon': 0.001,
    # force in-place updates of mean and variance estimates
    'updates_collections': None,
}

# Discriminator network
logit_ = nets.discriminator(x, y, weight_decay)

loss_op = losses.mean_squared_error(y_, y)

tf.image_summary('x', x)
tf.image_summary('y', y)
tf.image_summary('y_', y_disp_)
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
