import tensorflow as tf

layers = tf.contrib.layers
arg_scope = tf.contrib.framework.arg_scope

def generator(x,
              num_layer,
              num_channels,
              weight_decay):
    with arg_scope([layers.conv2d],
                   kernel_size=[3, 3],
                   weights_regularizer=layers.l2_regularizer(weight_decay),
                   biases_regularizer=layers.l2_regularizer(weight_decay),
                   variables_collections=tf.GraphKeys.TRAINABLE_VARIABLES,
                   outputs_collections=tf.GraphKeys.ACTIVATIONS):
        net = x
        net = layers.conv2d(inputs=net,
                            num_outputs=num_channels,
                            scope='G/conv0')
        for i in range(1, num_layer):
            net_temp = layers.convolution2d(inputs=net,
                                            num_outputs=num_channels,
                                            stride=1,
                                            scope='G/conv' + str(i))
            net = tf.add(net, net_temp, name='G/conv' + str(i) + '/add')
        net = layers.conv2d(net,
                            num_outputs=3,
                            scope='G/conv' + str(num_layer),
                            activation_fn=None)
    return net


def discriminator(x,
                  y,
                  weight_decay,
                  dropout_keep_prob=0.8,
                  reuse=None,
                  is_training=True):
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
    }

    with arg_scope([layers.conv2d],
                   kernel_size=[3, 3],
                   weights_regularizer=layers.l2_regularizer(weight_decay),
                   normalizer_fn=layers.batch_norm,
                   normalizer_params=batch_norm_params,
                   biases_regularizer=layers.l2_regularizer(weight_decay),
                   variables_collections=tf.GraphKeys.TRAINABLE_VARIABLES,
                   outputs_collections=tf.GraphKeys.ACTIVATIONS,
                   trainable=is_training):
        net = layers.conv2d(x, 32, stride=2, scope='D/conv_1a')
        net = layers.conv2d(net, 32, scope='D/conv_1b')
        net = layers.conv2d(net, 64, scope='D/conv_1c')
        net = layers.conv2d(net, 64, stride=2, scope='D/conv_2a')
        net = layers.conv2d(net, 64, scope='D/conv_2b')
        net = layers.conv2d(net, 128, scope='D/conv_2c')
        net = layers.conv2d(net, 128, stride=2, scope='D/conv_3a')
        net = layers.conv2d(net, 128, scope='D/conv_3b')
        net = layers.conv2d(net, 256, scope='D/conv_3c')
        net = layers.conv2d(net, 256, stride=3, scope='D/conv_4a')
        net = layers.conv2d(net, 256, scope='D/conv_4b')
        net = layers.conv2d(net, 256, scope='D/conv_4c')
        net = layers.conv2d(net, 256, scope='D/conv_5a')
        net = layers.conv2d(net, 256, scope='D/conv_5b')
        net = layers.conv2d(net, 256, scope='D/conv_5c')
    net = layers.flatten(net)
    net = layers.fully_connected(inputs=net,
                                 num_outputs=1024,
                                 normalizer_fn=layers.batch_norm,
                                 normalizer_params=batch_norm_params,
                                 weights_regularizer=layers.l2_regularizer(weight_decay),
                                 biases_regularizer = layers.l2_regularizer(weight_decay),
                                 variables_collections=tf.GraphKeys.TRAINABLE_VARIABLES,
                                 outputs_collections=tf.GraphKeys.ACTIVATIONS,
                                 trainable=is_training,
                                 scope='D/fc')
    net = layers.dropout(net, dropout_keep_prob,
                       is_training=is_training,
                       scope='D/dropout')
    net = tf.sigmoid(net)
    return net