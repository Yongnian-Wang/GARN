from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers
def lrelu(x, trainbable=None):
    return tf.maximum(x * 0.2, x)
def gam_block(input_feature, name, ratio):
    with tf.variable_scope(name):
        attention_feature = channel_attention(input_feature, 'ch_at', ratio)
        attention_feature = spatial_attention(attention_feature, 'sp_at', ratio)
        print("GAM Hello")
    return attention_feature
def channel_attention(input_feature, name, ratio):
    with tf.variable_scope(name):
        channel = input_feature.get_shape()[-1]
        batch = input_feature.get_shape()[0]
        view = input_feature.view(batch,,channel)
        Linear1 = tf.layers.dense(inputs=view,
                                  units=channel // ratio,
                                  activation=None,
                                  name='Linear1')
        Linear2 = tf.layers.dense(inputs=Linear1,
                                  units=channel,
                                  activation=None,
                                  name='Linear2')

        relu1 = tf.nn.relu(Linear2, name='relu1')
    return input_feature * relu1
def spatial_attention(input_feature, name, ratio):
    kernel_size = 8
    with tf.variable_scope(name):
        channel = input_feature.get_shape()[-1]
        conv1 = tf.layers.conv2d(input_feature,
                                 filters=channel // ratio,
                                 kernel_size=[kernel_size, kernel_size],
                                 strides=[1, 1],
                                 padding="same",
                                 activation=None,
                                 name='conv1_0')
        bn1 = tf.layers.batch_normalization(conv1, momentum=0.99, epsilon=0.001, training=True, name='bn1')
        relu2 = tf.nn.relu(bn1, name='relu2')
        conv2 = tf.layers.conv2d(relu2,
                                 filters=channel,
                                 kernel_size=[kernel_size, kernel_size],
                                 strides=[1, 1],
                                 padding="same",
                                 activation=None,
                                 name='conv1_1')
        bn2 = tf.layers.batch_normalization(conv2, momentum=0.99, epsilon=0.001, training=True, name='bn2')
        concat = tf.sigmoid(bn2, 'sigmoid0_1')
    return input_feature * concat
def upsample_and_concat(x1, x2, output_channels, in_channels, scope_name, trainable=True):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        pool_size = 2
        deconv_filter = tf.get_variable('weights', [pool_size, pool_size, output_channels, in_channels], trainable=True)
        deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1],
                                        name=scope_name)

        deconv_output = tf.concat([deconv, x2], 3)
        deconv_output.set_shape([None, None, None, output_channels * 2])
        return deconv_output
def DecomNet_simple(input):
    with tf.variable_scope('DecomNet', reuse=tf.AUTO_REUSE):
        conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_1')
        pool1 = slim.max_pool2d(conv1, [2, 2], stride=2, padding='SAME')
        conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_1')
        pool2 = slim.max_pool2d(conv2, [2, 2], stride=2, padding='SAME')
        conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_1')
        up8 = upsample_and_concat(conv3, conv2, 64, 128, 'g_up_1')
        conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_1')
        up9 = upsample_and_concat(conv8, conv1, 32, 64, 'g_up_2')
        conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_1')
        conv10 = slim.conv2d(conv9, 3, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
        R_out = tf.sigmoid(conv10)
        l_conv2 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='l_conv1_2')
        l_conv3 = tf.concat([l_conv2, conv9], 3)
        l_conv4 = slim.conv2d(l_conv3, 1, [1, 1], rate=1, activation_fn=None, scope='l_conv1_4')
        L_out = tf.sigmoid(l_conv4)
    return R_out, L_out
def Restoration_net(input_r, input_i):
    with tf.variable_scope('Restoration_net', reuse=tf.AUTO_REUSE):
        conv0_0 = slim.conv2d(input_r, 32, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='de_conv0_0')
        attention0 = gam_block(conv0_0, name='de_conv0', ratio=4)
        conv0_1 = slim.conv2d(input_i, 4, [3, 3], rate=1, activation_fn=tf.nn.sigmoid, scope='de_conv0_1')
        attention = spatial_attention1(conv0_1, name='de_conv0_2', ratio=4)
        input_all = tf.concat([attention0, attention], 3)
        conv1 = slim.conv2d(input_all, 32, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv1_1')
        conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv1_2')
        attention1 = gam_block(conv1, name='de_conv1', ratio=4)
        pool1 = slim.max_pool2d(attention1, [2, 2], padding='SAME')
        conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv2_1')
        conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv2_2')
        attention2 = gam_block(conv2, name='de_conv2', ratio=4)
        pool2 = slim.max_pool2d(attention2, [2, 2], padding='SAME')
        conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv3_1')
        conv3 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv3_2')
        attention3 = gam_block(conv3, name='de_conv3', ratio=4)
        pool3 = slim.max_pool2d(attention3, [2, 2], padding='SAME')
        conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv4_1')
        conv4 = slim.conv2d(conv4, 256, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv4_2')
        attention4 = gam_block(conv4, name='de_conv4', ratio=4)
        pool4 = slim.max_pool2d(attention4, [2, 2], padding='SAME')
        conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv5_1')
        conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv5_2')
        attention5 = gam_block(conv5, name='de_conv5', ratio=4)
        up6 = upsample_and_concat(attention5, conv4, 256, 512, 'up_6')
        conv6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv6_1')
        conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv6_2')
        attention6 = gam_block(conv6, name='de_conv6', ratio=4)
        up7 = upsample_and_concat(attention6, conv3, 128, 256, 'up_7')
        conv7 = slim.conv2d(up7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv7_1')
        conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv7_2')
        attention7 = gam_block(conv7, name='de_conv7', ratio=4)
        up8 = upsample_and_concat(attention7, conv2, 64, 128, 'up_8')
        conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv8_1')
        conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv8_2')
        attention8 = gam_block(conv8, name='de_conv8', ratio=4)
        up9 = upsample_and_concat(attention8, conv1, 32, 64, 'up_9')
        conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv9_1')
        conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv9_2')
        attention9 = gam_block(conv9, name='de_conv9', ratio=4)
        conv10 = slim.conv2d(attention9, 3, [3, 3], rate=1, activation_fn=None, scope='de_conv10')
        out = tf.sigmoid(conv10)
        return out
def Illumination_adjust_net(input_i, input_ratio):
    with tf.variable_scope('Illumination_adjust_net', reuse=tf.AUTO_REUSE):
        input_all = tf.concat([input_i, input_ratio], 3)
        conv1 = slim.conv2d(input_all, 32, [3, 3], rate=1, activation_fn=lrelu, scope='en_conv_1')
        conv2 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='en_conv_2')
        conv3 = slim.conv2d(conv2, 32, [3, 3], rate=1, activation_fn=lrelu, scope='en_conv_3')
        conv4 = slim.conv2d(conv3, 1, [3, 3], rate=1, activation_fn=lrelu, scope='en_conv_4')
        L_enhance = tf.sigmoid(conv4)
    return L_enhance