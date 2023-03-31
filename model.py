from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers

def lrelu(x, trainbable=None):
    return tf.maximum(x * 0.2, x)

def se_block(residual, name, ratio=8):
    """Contains the implementation of Squeeze-and-Excitation(SE) block.
    As described in https://arxiv.org/abs/1709.01507.
    """

    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)

    with tf.variable_scope(name):
        channel = residual.get_shape()[-1]
        # Global average pooling
        squeeze = tf.reduce_mean(residual, axis=[1, 2], keepdims=True)
        assert squeeze.get_shape()[1:] == (1, 1, channel)
        excitation = tf.layers.dense(inputs=squeeze,
                                     units=channel // ratio,
                                     activation=tf.nn.relu,
                                     kernel_initializer=kernel_initializer,
                                     bias_initializer=bias_initializer,
                                     name='bottleneck_fc')
        assert excitation.get_shape()[1:] == (1, 1, channel // ratio)
        excitation = tf.layers.dense(inputs=excitation,
                                     units=channel,
                                     activation=tf.nn.sigmoid,
                                     kernel_initializer=kernel_initializer,
                                     bias_initializer=bias_initializer,
                                     name='recover_fc')
        assert excitation.get_shape()[1:] == (1, 1, channel)
        # top = tf.multiply(bottom, se, name='scale')
        scale = residual * excitation
    return scale

def cbam_block(input_feature, name, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """

    with tf.variable_scope(name):
        attention_feature = channel_attention(input_feature, 'ch_at', ratio)
        attention_feature = spatial_attention(attention_feature, 'sp_at')
        print("CBAM Hello")
    return attention_feature

def channel_attention(input_feature, name, ratio):
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)

    with tf.variable_scope(name):
        # 获得tensor的形状，最后一个维度
        channel = input_feature.get_shape()[-1]
        # keepdims = false 减少一个维度 / keepdims = true 保持不变
        avg_pool = tf.reduce_mean(input_feature, axis=[1, 2], keepdims=True)

        assert avg_pool.get_shape()[1:] == (1, 1, channel)
        # dense添加一层全连接层
        # ratio为缩减比，减少模型参数量
        # MLP多层感知器，神经元为神经网络的计算单元，包括加权、偏执、激活函数
        avg_pool = tf.layers.dense(inputs=avg_pool,
                                   # units 输出维度大小=隐藏层神经元的个数
                                   units=channel // ratio,
                                   activation=tf.nn.relu,
                                   kernel_initializer=kernel_initializer,
                                   bias_initializer=bias_initializer,
                                   name='mlp_0',
                                   reuse=None)
        assert avg_pool.get_shape()[1:] == (1, 1, channel // ratio)
        avg_pool = tf.layers.dense(inputs=avg_pool,
                                   units=channel,
                                   kernel_initializer=kernel_initializer,
                                   bias_initializer=bias_initializer,
                                   name='mlp_1',
                                   reuse=None)
        assert avg_pool.get_shape()[1:] == (1, 1, channel)

        max_pool = tf.reduce_max(input_feature, axis=[1, 2], keepdims=True)
        assert max_pool.get_shape()[1:] == (1, 1, channel)
        max_pool = tf.layers.dense(inputs=max_pool,
                                   units=channel // ratio,
                                   activation=tf.nn.relu,
                                   name='mlp_0',
                                   reuse=True)
        assert max_pool.get_shape()[1:] == (1, 1, channel // ratio)
        max_pool = tf.layers.dense(inputs=max_pool,
                                   units=channel,
                                   name='mlp_1',
                                   reuse=True)
        assert max_pool.get_shape()[1:] == (1, 1, channel)

        scale = tf.sigmoid(avg_pool + max_pool, 'sigmoid0_0')

    return input_feature * scale

def spatial_attention(input_feature, name):
    kernel_size = 8
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    with tf.variable_scope(name):
        avg_pool = tf.reduce_mean(input_feature, axis=[3], keepdims=True)
        assert avg_pool.get_shape()[-1] == 1
        max_pool = tf.reduce_max(input_feature, axis=[3], keepdims=True)
        assert max_pool.get_shape()[-1] == 1
        concat = tf.concat([avg_pool, max_pool], 3)
        assert concat.get_shape()[-1] == 2

        concat = tf.layers.conv2d(concat,
                                  filters=1,
                                  kernel_size=[kernel_size, kernel_size],
                                  strides=[1, 1],
                                  padding="same",
                                  activation=None,
                                  kernel_initializer=kernel_initializer,
                                  use_bias=False,
                                  name='conv0_0')
        assert concat.get_shape()[-1] == 1
        concat = tf.sigmoid(concat, 'sigmoid0_0')

    return input_feature * concat

def gam_block(input_feature, name, ratio):
    with tf.variable_scope(name):
        # tensor.shape(batch, height, width, channel)
        # b, h, w, c = input_feature.get_shape
        # input_feature_permute = input_feature.permute
        attention_feature = channel_attention1(input_feature, 'ch_at1', ratio)
        attention_feature = spatial_attention1(attention_feature, 'sp_at1', ratio)
        # attention_feature = spatial_attention(attention_feature, 'sp_at1')
        print("GAM Hello")
    return attention_feature

def channel_attention1(input_feature, name, ratio):
    # kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    # bias_initializer = tf.constant_initializer()
    # b, h, w, w = input_feature.get_shape()
    with tf.variable_scope(name):
        # 获得tensor的形状，最后一个维度
        channel = input_feature.get_shape()[-1]

        # dense添加一层全连接层
        # ratio为缩减比，减少模型参数量
        # MLP多层感知器，神经元为神经网络的计算单元，包括加权、偏执、激活函数
        Linear1 = tf.layers.dense(inputs=input_feature,
                                  units=channel // ratio,
                                  # units=channel,
                                  activation=None,
                                  # kernel_initializer=kernel_initializer,
                                  # bias_initializer=bias_initializer,
                                  name='Linear1')
        # relu1 = tf.nn.relu(Linear1, name='relu1')
        Linear2 = tf.layers.dense(inputs=Linear1,
                                  units=channel,
                                  activation=None,
                                  # kernel_initializer=kernel_initializer,
                                  # bias_initializer=bias_initializer,
                                  name='Linear2')

        relu1 = tf.nn.relu(Linear2, name='relu1')
        # p = Linear2.view(b, h, w, c)
    # return input_feature * p
    # return input_feature * Linear2
    return input_feature * relu1

def spatial_attention1(input_feature, name, ratio):
    # kernel_size = 7
    kernel_size = 8
    # kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    # bias_initializer = tf.constant_initializer()
    with tf.variable_scope(name):
        # filters：整数,输出空间的维数(即卷积中的滤波器数).
        # kernel_size：2个整数的整数或元组/列表,指定2D卷积窗口的高度和宽度.可以是单个整数,以指定所有空间维度的相同值.
        # strides：2个整数的整数或元组/列表,指定卷积沿高度和宽度的跨度.可以是单个整数,以指定所有空间维度的相同值.指定任何步幅值！= 1与指定任何dilation_rate值！= 1都不相容.
        # padding：可以是"valid"或"same"(不区分大小写).
        # data_format：一个字符串,可以是channels_last(默认)或channels_first,表示输入中维度的顺序,channels_last对应于具有形状(batch, height, width, channels)的输入,
        # 而channels_first对应于具有形状(batch, channels, height, width)的输入.
        # dilation_rate：2个整数的整数或元组/列表,指定用于扩张卷积的扩张率.可以是单个整数,以指定所有空间维度的相同值.目前,指定任何dilation_rate值！= 1与指定任何步幅值！= 1都不相容.
        # activation：激活功能,将其设置为“None”以保持线性激活.
        # use_bias：Boolean,该层是否使用偏差.
        # kernel_initializer：卷积内核的初始化程序.
        # bias_initializer：偏置向量的初始化器,如果为None,将使用默认初始值设定项.
        # kernel_regularizer：卷积内核的可选正则化器.
        # bias_regularizer：偏置矢量的可选正则化器.
        # activity_regularizer：输出的可选正则化函数.
        # kernel_constraint：由Optimizer更新后应用于内核的可选投影函数(例如,用于实现层权重的范数约束或值约束).该函数必须将未投影的变量作为输入,并且必须返回投影变量(必须具有相同的形状).在进行异步分布式培训时,使用约束是不安全的.
        # bias_constraint：由Optimizer更新后应用于偏差的可选投影函数.
        # trainable：Boolean,如果为True,还将变量添加到图集合GraphKeys.TRAINABLE_VARIABLES中(请参阅参考资料tf.Variable).
        # name：字符串,图层的名称.
        channel = input_feature.get_shape()[-1]
        conv1 = tf.layers.conv2d(input_feature,
                                 filters=channel // ratio,
                                 kernel_size=[kernel_size, kernel_size],
                                 strides=[1, 1],
                                 padding="same",
                                 activation=None,
                                 # kernel_initializer=kernel_initializer,
                                 # bias_initializer=bias_initializer,
                                 name='conv1_0')
        # tf.layers.batch_normalization和tf.contrib.layers.batch_norm可以用来构建待训练的神经网络模型，而tf.nn.batch_normalization一般只用来构建推理模型。
        # tf.layers.batch_normalization(
        #     inputs,
        #     axis=-1,axis：默认值是-1，也就是说默认的是最后一个维度，这个参数的意思是批标准化处理的维度是以最后一个维度进行的，也就是channel，当然你也可以改（万一有奇迹呢，AI有时候就是不按常理出牌）
        #     momentum=0.99,值用在训练时，滑动平均的方式计算滑动平均值moving_mean和滑动方差moving_variance。
        #     epsilon=0.001,
        #     center=True,为True时，添加位移因子beta到该BN层，否则不添加。添加beta是对BN层的变换加入位移操作。注意，beta一般设定为可训练参数，即trainable=True。
        #     scale=True,为True是，添加缩放因子gamma到该BN层，否则不添加。添加gamma是对BN层的变化加入缩放操作。注意，gamma一般设定为可训练参数，即trainable=True。
        #     beta_initializer=tf.zeros_initializer(),
        #     gamma_initializer=tf.ones_initializer(),
        #     moving_mean_initializer=tf.zeros_initializer(),
        #     moving_variance_initializer=tf.ones_initializer(),
        #     beta_regularizer=None,
        #     gamma_regularizer=None,
        #     beta_constraint=None,
        #     gamma_constraint=None,
        #     training=False,表示模型当前的模式，如果为True，则模型在训练模式，否则为推理模式。要非常注意这个模式的设定，这个参数默认值为False。如果在训练时采用了默认值False，则滑动均值moving_mean和滑动方差moving_variance都不会根据当前batch的数据更新，这就意味着在推理模式下，均值和方差都是其初始值，因为这两个值并没有在训练迭代过程中滑动更新。
        #     trainable=True,
        #     name=None,
        #     reuse=None,
        #     renorm=False,
        #     renorm_clipping=None,
        #     renorm_momentum=0.99,
        #     fused=None,
        #     virtual_batch_size=None,
        #     adjustment=None
        # )
        bn1 = tf.layers.batch_normalization(conv1, momentum=0.99, epsilon=0.001, training=True, name='bn1')
        relu2 = tf.nn.relu(bn1, name='relu2')
        conv2 = tf.layers.conv2d(relu2,
                                 filters=channel,
                                 kernel_size=[kernel_size, kernel_size],
                                 strides=[1, 1],
                                 padding="same",
                                 activation=None,
                                 # kernel_initializer=kernel_initializer,
                                 # bias_initializer=bias_initializer,
                                 name='conv1_1')
        bn2 = tf.layers.batch_normalization(conv2, momentum=0.99, epsilon=0.001, training=True, name='bn2')
        concat = tf.sigmoid(bn2, 'sigmoid0_1')

    return input_feature * concat

def residual_block(residual_input, transient_input, n_filters):
    # 没加作用域
    # with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope
    # with tf.variable_scope(name):
    residual_stram_main = tf.layers.conv2d(residual_input, filters=n_filters, kernel_size=[3, 3], strides=[1, 1], padding="same", activation=None)
    residual_stram_extra = tf.layers.conv2d(residual_input, filters=n_filters, kernel_size=[3, 3], strides=[1, 1], padding="same", activation=None)
    transient_stream_main = tf.layers.conv2d(transient_input, filters=n_filters, kernel_size=[3, 3], strides=[1, 1], padding="same", activation=None)
    transient_stream_extra = tf.layers.conv2d(transient_input, filters=n_filters, kernel_size=[3, 3], strides=[1, 1], padding="same", activation=None)
    # add1
    residual_stram_main = tf.add(residual_stram_main, transient_stream_extra)
    # add2
    residual_stram_main = tf.add(residual_stram_main, residual_input)
    # add3
    transient_stream_main = tf.add(transient_stream_main, residual_stram_extra)
    # residual_BN
    residual_stram_main = tf.layers.batch_normalization(residual_stram_main, momentum=0.99, epsilon=0.001, training=True)
    residual_stram_main = tf.nn.relu(residual_stram_main)
    # transient_BN
    transient_stream_main = tf.layers.batch_normalization(transient_stream_main, momentum=0.99, epsilon=0.001, training=True)
    transient_stream_main = tf.nn.relu(transient_stream_main)
    print("residual_block")
    return residual_stram_main, transient_stream_main

def upsample_and_concat(x1, x2, output_channels, in_channels, scope_name, trainable=True):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        pool_size = 2
        # tf.get_variable定义共享变量
        deconv_filter = tf.get_variable('weights', [pool_size, pool_size, output_channels, in_channels], trainable=True)
        # conv2d_transpose(x,filter,output_shape,strides,padding='SAME',data_format='NHWC',name=None)
        # x(shape(batch_size,w,h,channel))
        # filter: [kernel_size,kernel_size, output_channels, input_channels]
        # stride (batch_size,w,h,channel)像素填充步长
        # 将 inputs 进行填充扩大，扩大的倍数与strides有关（strides倍）。扩大的方式是在元素之间插[ strides - 1 ]个 0。
        # padding="VALID"时，在插完值后继续在周围填充的宽度为[ kenel_size - 1 ],填充值为0；padding = "SAME"时，在插完值后根据output尺寸进行填充,填充值为0。
        # 对扩充变大的矩阵，用大小为kernel_size卷积核做卷积操作，这样的卷积核有filters个，并且这里的步长为1(与参数strides无关，一定是1)
        deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1],
                                        name=scope_name)
        deconv_output = tf.concat([deconv, x2], 3)
        deconv_output.set_shape([None, None, None, output_channels * 2])
        return deconv_output

def DecomNet_simple(input):
    # tf.variable_scope共享变量
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
        # Here, we use 1*1 kernel to replace the 3*3 ones in the paper to get better results.
        conv10 = slim.conv2d(conv9, 3, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
        R_out = tf.sigmoid(conv10)

        l_conv2 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='l_conv1_2')
        l_conv3 = tf.concat([l_conv2, conv9], 3)
        # Here, we use 1*1 kernel to replace the 3*3 ones in the paper to get better results.
        l_conv4 = slim.conv2d(l_conv3, 1, [1, 1], rate=1, activation_fn=None, scope='l_conv1_4')
        L_out = tf.sigmoid(l_conv4)

    return R_out, L_out

# RIR
# def Restoration_net(input_r, input_i):
#     with tf.variable_scope('Restoration_net', reuse=tf.AUTO_REUSE):
#         # # 给反射图像引入attention
#         # conv0_0 = slim.conv2d(input_r, 32, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='de_conv0_0')
#         # attention0 = gam_block(conv0_0, name='de_conv0', ratio=4)
#         # # 亮度图像引入attention
#         conv0_1 = slim.conv2d(input_i, 32, [3, 3], rate=1, activation_fn=tf.nn.sigmoid, scope='de_conv0_1')
#         conv0_2 = slim.conv2d(input_i, 64, [3, 3], rate=1, activation_fn=tf.nn.sigmoid, scope='de_conv0_2')
#         conv0_3 = slim.conv2d(input_i, 128, [3, 3], rate=1, activation_fn=tf.nn.sigmoid, scope='de_conv0_3')
#         conv0_4 = slim.conv2d(input_i, 256, [3, 3], rate=1, activation_fn=tf.nn.sigmoid, scope='de_conv0_4')
#         conv0_5 = slim.conv2d(input_i, 512, [3, 3], rate=1, activation_fn=tf.nn.sigmoid, scope='de_conv0_5')
#         # attention = spatial_attention1(conv0_1, name='de_conv0_2', ratio=4)
#         # # 合并
#         # input_all = tf.concat([attention0, attention], 3)
#
#         conv1 = slim.conv2d(input_r, 32, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv1_1')
#         conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv1_2')
#         illumination1, reflectance1 = residual_block(conv0_1, conv1, 32, name='residual_block_1_1')
#         illumination1, reflectance1 = residual_block(illumination1, reflectance1, 32, name='residual_block_1_2')
#         merge1 = tf.concat([illumination1, reflectance1], 3)
#
#         conv2 = slim.conv2d(merge1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv2_1')
#         conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv2_2')
#         illumination2, reflectance2 = residual_block(conv0_2, conv2, 64, name='residual_block_2_1')
#         illumination2, reflectance2 = residual_block(illumination2, reflectance2, 64, name='residual_block_2_2')
#         merge2 = tf.concat([illumination2, reflectance2], 3)
#
#         conv3 = slim.conv2d(merge2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv3_1')
#         conv3 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv3_2')
#         illumination3, reflectance3 = residual_block(conv0_3, conv3, 128, name='residual_block_3_1')
#         illumination3, reflectance3 = residual_block(illumination3, reflectance3, 128, name='residual_block_3_2')
#         merge3 = tf.concat([illumination3, reflectance3], 3)
#
#         conv4 = slim.conv2d(merge3, 256, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv4_1')
#         conv4 = slim.conv2d(conv4, 256, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv4_2')
#         illumination4, reflectance4 = residual_block(conv0_4, conv4, 256, name='residual_block_4_1')
#         illumination4, reflectance4 = residual_block(illumination4, reflectance4, 256, name='residual_block_4_2')
#         merge4 = tf.concat([illumination4, reflectance4], 3)
#
#         conv5 = slim.conv2d(merge4, 512, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv5_1')
#         conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv5_2')
#         illumination5, reflectance5 = residual_block(conv0_5, conv5, 512, name='residual_block_5_1')
#         illumination5, reflectance5 = residual_block(illumination5, reflectance5, 512, name='residual_block_5_2')
#         merge5 = tf.concat([illumination5, reflectance5], 3)
#
#         conv10 = slim.conv2d(merge5, 3, [3, 3], rate=1, activation_fn=None, scope='de_conv10')
#
#         out = tf.sigmoid(conv10)
#         return out

# ill conv32(3x3), refl conv32(3x3)
# 2 * RIR(ill, ref)
# GAM_block(ill), GAM_block(ref), concat(ill,refl)
def Restoration_net(input_r, input_i):
    with tf.variable_scope('Restoration_net', reuse=tf.AUTO_REUSE):
        # 反射图像和照度图像经过Conv改变成相同的通道数
        conv0_0 = slim.conv2d(input_r, 32, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='de_conv0_0')
        conv0_1 = slim.conv2d(input_i, 32, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='de_conv0_1')
        # RIR
        illumination0_1, reflectance0_1 = residual_block(conv0_1, conv0_0, 32)
        illumination0_2, reflectance0_2 = residual_block(illumination0_1, reflectance0_1, 32)
        # 给反射图像引入attention
        attention0 = gam_block(reflectance0_2, name='gamr_0', ratio=4)
        # 亮度图像引入attention
        attention = gam_block(illumination0_2, name='gami_0', ratio=4)

        # 合并
        input_all = tf.concat([attention0, attention], 3)
        # input_all = tf.concat([input_r, input_i], 3)

        conv1 = slim.conv2d(input_all, 32, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv1_1')
        conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv1_2')
        # attention1 = cbam_block(conv1, name='de_conv1', ratio=8)
        attention1 = gam_block(conv1, name='de_conv1', ratio=4)
        pool1 = slim.max_pool2d(attention1, [2, 2], padding='SAME')
        # pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')

        conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv2_1')
        conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv2_2')
        # attention2 = cbam_block(conv2, name='de_conv2', ratio=8)
        attention2 = gam_block(conv2, name='de_conv2', ratio=4)
        pool2 = slim.max_pool2d(attention2, [2, 2], padding='SAME')
        # pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')

        conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv3_1')
        conv3 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv3_2')
        # attention3 = cbam_block(conv3, name='de_conv3', ratio=8)
        attention3 = gam_block(conv3, name='de_conv3', ratio=4)
        pool3 = slim.max_pool2d(attention3, [2, 2], padding='SAME')
        # pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

        conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv4_1')
        conv4 = slim.conv2d(conv4, 256, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv4_2')
        # attention4 = cbam_block(conv4, name='de_conv4', ratio=8)
        attention4 = gam_block(conv4, name='de_conv4', ratio=4)
        pool4 = slim.max_pool2d(attention4, [2, 2], padding='SAME')
        # pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')

        conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv5_1')
        conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv5_2')
        # attention5 = cbam_block(conv5, name='de_conv5', ratio=8)
        attention5 = gam_block(conv5, name='de_conv5', ratio=4)

        # up6 = upsample_and_concat(conv5, conv4, 256, 512, 'up_6')
        up6 = upsample_and_concat(attention5, conv4, 256, 512, 'up_6')
        conv6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv6_1')
        conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv6_2')
        # attention6 = cbam_block(conv6, name='de_conv6', ratio=8)
        attention6 = gam_block(conv6, name='de_conv6', ratio=4)

        # up7 = upsample_and_concat(conv6, conv3, 128, 256, 'up_7')
        up7 = upsample_and_concat(attention6, conv3, 128, 256, 'up_7')
        conv7 = slim.conv2d(up7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv7_1')
        conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv7_2')
        # attention7 = cbam_block(conv7, name='de_conv7', ratio=8)
        attention7 = gam_block(conv7, name='de_conv7', ratio=4)

        # up8 = upsample_and_concat(conv7, conv2, 64, 128, 'up_8')
        up8 = upsample_and_concat(attention7, conv2, 64, 128, 'up_8')
        conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv8_1')
        conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv8_2')
        # attention8 = cbam_block(conv8, name='de_conv8', ratio=8)
        attention8 = gam_block(conv8, name='de_conv8', ratio=4)

        # up9 = upsample_and_concat(conv8, conv1, 32, 64, 'up_9')
        up9 = upsample_and_concat(attention8, conv1, 32, 64, 'up_9')
        conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv9_1')
        conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv9_2')
        # attention9 = cbam_block(conv9, name='de_conv9', ratio=8)
        attention9 = gam_block(conv9, name='de_conv9', ratio=4)

        # conv10 = slim.conv2d(conv9, 3, [3, 3], rate=1, activation_fn=None, scope='de_conv10')
        conv10 = slim.conv2d(attention9, 3, [3, 3], rate=1, activation_fn=None, scope='de_conv10')

        out = tf.sigmoid(conv10)
        return out

# ill conv4(3x3)-sigmoid-SPattention1
# refl conv32(3x3)-relu-GAM_block
# concat(ill,refl)-GAM_block
# def Restoration_net(input_r, input_i):
#     with tf.variable_scope('Restoration_net', reuse=tf.AUTO_REUSE):
#         # 给反射图像引入attention
#         conv0_0 = slim.conv2d(input_r, 32, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='de_conv0_0')
#         attention0 = gam_block(conv0_0, name='de_conv0', ratio=4)
#         # 亮度图像引入attention
#         conv0_1 = slim.conv2d(input_i, 4, [3, 3], rate=1, activation_fn=tf.nn.sigmoid, scope='de_conv0_1')
#         attention = spatial_attention1(conv0_1, name='de_conv0_2', ratio=4)
#         # 合并
#         input_all = tf.concat([attention0, attention], 3)
#         # input_all = tf.concat([input_r, input_i], 3)
#
#         conv1 = slim.conv2d(input_all, 32, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv1_1')
#         conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv1_2')
#         # attention1 = cbam_block(conv1, name='de_conv1', ratio=8)
#         attention1 = gam_block(conv1, name='de_conv1', ratio=4)
#         pool1 = slim.max_pool2d(attention1, [2, 2], padding='SAME')
#         # pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')
#
#         conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv2_1')
#         conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv2_2')
#         # attention2 = cbam_block(conv2, name='de_conv2', ratio=8)
#         attention2 = gam_block(conv2, name='de_conv2', ratio=4)
#         pool2 = slim.max_pool2d(attention2, [2, 2], padding='SAME')
#         # pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')
#
#         conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv3_1')
#         conv3 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv3_2')
#         # attention3 = cbam_block(conv3, name='de_conv3', ratio=8)
#         attention3 = gam_block(conv3, name='de_conv3', ratio=4)
#         pool3 = slim.max_pool2d(attention3, [2, 2], padding='SAME')
#         # pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')
#
#         conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv4_1')
#         conv4 = slim.conv2d(conv4, 256, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv4_2')
#         # attention4 = cbam_block(conv4, name='de_conv4', ratio=8)
#         attention4 = gam_block(conv4, name='de_conv4', ratio=4)
#         pool4 = slim.max_pool2d(attention4, [2, 2], padding='SAME')
#         # pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')
#
#         conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv5_1')
#         conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv5_2')
#         # attention5 = cbam_block(conv5, name='de_conv5', ratio=8)
#         attention5 = gam_block(conv5, name='de_conv5', ratio=4)
#
#         # up6 = upsample_and_concat(conv5, conv4, 256, 512, 'up_6')
#         up6 = upsample_and_concat(attention5, conv4, 256, 512, 'up_6')
#         conv6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv6_1')
#         conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv6_2')
#         # attention6 = cbam_block(conv6, name='de_conv6', ratio=8)
#         attention6 = gam_block(conv6, name='de_conv6', ratio=4)
#
#         # up7 = upsample_and_concat(conv6, conv3, 128, 256, 'up_7')
#         up7 = upsample_and_concat(attention6, conv3, 128, 256, 'up_7')
#         conv7 = slim.conv2d(up7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv7_1')
#         conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv7_2')
#         # attention7 = cbam_block(conv7, name='de_conv7', ratio=8)
#         attention7 = gam_block(conv7, name='de_conv7', ratio=4)
#
#         # up8 = upsample_and_concat(conv7, conv2, 64, 128, 'up_8')
#         up8 = upsample_and_concat(attention7, conv2, 64, 128, 'up_8')
#         conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv8_1')
#         conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv8_2')
#         # attention8 = cbam_block(conv8, name='de_conv8', ratio=8)
#         attention8 = gam_block(conv8, name='de_conv8', ratio=4)
#
#         # up9 = upsample_and_concat(conv8, conv1, 32, 64, 'up_9')
#         up9 = upsample_and_concat(attention8, conv1, 32, 64, 'up_9')
#         conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv9_1')
#         conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv9_2')
#         # attention9 = cbam_block(conv9, name='de_conv9', ratio=8)
#         attention9 = gam_block(conv9, name='de_conv9', ratio=4)
#
#         # conv10 = slim.conv2d(conv9, 3, [3, 3], rate=1, activation_fn=None, scope='de_conv10')
#         conv10 = slim.conv2d(attention9, 3, [3, 3], rate=1, activation_fn=None, scope='de_conv10')
#
#         out = tf.sigmoid(conv10)
#         return out

# refl conv32(3x3)-SP_attention
# concat(ill,refl)-CBAM_block
# def Restoration_net(input_r, input_i):
#     with tf.variable_scope('Restoration_net', reuse=tf.AUTO_REUSE):
#         # 给反射图像引入attention
#         conv0_0 = slim.conv2d(input_r, 32, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='de_conv0_0')
#         # attention0 = se_block(conv0_0, name='de_conv0', ratio=8)
#         attention0 = cbam_block(conv0_0, name='de_conv0', ratio=8)
#         # 给照度图像引入attention
#         conv0_1 = slim.conv2d(input_i, 4, [3, 3], rate=1, activation_fn=tf.nn.sigmoid, scope='de_conv0_1')
#         attention = spatial_attention(conv0_1, name='de_conv0_2')
#         # 将输入的反射图像与亮度图像级联并缩小到3通道
#         # input_all = tf.concat([attention, input_i], 3)
#         input_all = tf.concat([attention0, attention], 3)
#
#         conv1 = slim.conv2d(input_all, 32, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv1_1')
#         conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv1_2')
#         attention1 = cbam_block(conv1, name='de_conv1', ratio=8)
#         pool1 = slim.max_pool2d(attention1, [2, 2], padding='SAME')
#
#         conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv2_1')
#         conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv2_2')
#         attention2 = cbam_block(conv2, name='de_conv2', ratio=8)
#         pool2 = slim.max_pool2d(attention2, [2, 2], padding='SAME')
#
#         conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv3_1')
#         conv3 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv3_2')
#         attention3 = cbam_block(conv3, name='de_conv3', ratio=8)
#         pool3 = slim.max_pool2d(attention3, [2, 2], padding='SAME')
#
#         conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv4_1')
#         conv4 = slim.conv2d(conv4, 256, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv4_2')
#         attention4 = cbam_block(conv4, name='de_conv4', ratio=8)
#         pool4 = slim.max_pool2d(attention4, [2, 2], padding='SAME')
#
#         conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv5_1')
#         conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv5_2')
#         attention5 = cbam_block(conv5, name='de_conv5', ratio=8)
#
#
#         up6 = upsample_and_concat(attention5, conv4, 256, 512, 'up_6')
#         conv6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv6_1')
#         conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv6_2')
#         attention6 = cbam_block(conv6, name='de_conv6', ratio=8)
#
#
#         up7 = upsample_and_concat(attention6, conv3, 128, 256, 'up_7')
#         conv7 = slim.conv2d(up7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv7_1')
#         conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv7_2')
#         attention7 = cbam_block(conv7, name='de_conv7', ratio=8)
#
#
#         up8 = upsample_and_concat(attention7, conv2, 64, 128, 'up_8')
#         conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv8_1')
#         conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv8_2')
#         attention8 = cbam_block(conv8, name='de_conv8', ratio=8)
#
#
#         up9 = upsample_and_concat(attention8, conv1, 32, 64, 'up_9')
#         conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv9_1')
#         conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv9_2')
#         attention9 = cbam_block(conv9, name='de_conv9', ratio=8)
#
#
#         conv10 = slim.conv2d(attention9, 3, [3, 3], rate=1, activation_fn=None, scope='de_conv10')
#
#         out = tf.sigmoid(conv10)
#         return out

# def Illumination_adjust_net(input_i, input_ratio):
#     with tf.variable_scope('Illumination_adjust_net', reuse=tf.AUTO_REUSE):
#         input_all = tf.concat([input_i, input_ratio], 3)
#
#         conv1 = slim.conv2d(input_all, 32, [3, 3], rate=1, activation_fn=lrelu, scope='en_conv_1')
#         conv2 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='en_conv_2')
#         conv3 = slim.conv2d(conv2, 32, [3, 3], rate=1, activation_fn=lrelu, scope='en_conv_3')
#         conv4 = slim.conv2d(conv3, 1, [3, 3], rate=1, activation_fn=lrelu, scope='en_conv_4')
#
#         L_enhance = tf.sigmoid(conv4)
#     return L_enhance

# 自适应曲线增强
def Illumination_adjust_net(input_i):
    with tf.variable_scope('Illumination_adjust_net', reuse=tf.AUTO_REUSE):
        conv1 = slim.conv2d(input_i, 16, [3, 3], rate=1, activation_fn=lrelu, scope='en_conv_1')
        conv2 = slim.conv2d(conv1, 16, [3, 3], rate=1, activation_fn=lrelu, scope='en_conv_2')
        conv3 = slim.conv2d(conv2, 16, [3, 3], rate=1, activation_fn=lrelu, scope='en_conv_3')
        conv4 = slim.conv2d(conv3, 16, [3, 3], rate=1, activation_fn=lrelu, scope='en_conv_4')
        conv4 = tf.concat([conv3, conv4], 3)
        conv5 = slim.conv2d(conv4, 16, [3, 3], rate=1, activation_fn=lrelu, scope='en_conv_5')
        conv5 = tf.concat([conv2, conv5], 3)
        conv6 = slim.conv2d(conv5, 16, [3, 3], rate=1, activation_fn=lrelu, scope='en_conv_6')
        conv6 = tf.concat([conv1, conv6], 3)
        conv7 = slim.conv2d(conv6, 4, [3, 3], rate=1, activation_fn=lrelu, scope='en_conv_7')

        r1 = conv7[:, :, :, 0:1]
        r2 = conv7[:, :, :, 1:2]
        r3 = conv7[:, :, :, 2:3]
        r4 = conv7[:, :, :, 3:4]
        r = tf.concat([r1, r2, r3, r4], 3)

        x1 = input_i + r1 * (input_i ** 2 - input_i)
        x2 = x1 + r2 * (x1 ** 2 - x1)
        x3 = x2 + r3 * (x2 ** 2 - x2)
        x4 = x3 + r4 * (x3 ** 2 - x3)
    return x4, r

# 源
# def Illumination_adjust_net(input_i, input_ratio):
#     with tf.variable_scope('Illumination_adjust_net', reuse=tf.AUTO_REUSE):
#         input_all = tf.concat([input_i, input_ratio], 3)
#
#         conv1 = slim.conv2d(input_all, 32, [3, 3], rate=1, activation_fn=lrelu, scope='en_conv_1')
#         conv2 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='en_conv_2')
#         conv3 = slim.conv2d(conv2, 32, [3, 3], rate=1, activation_fn=lrelu, scope='en_conv_3')
#         conv4 = slim.conv2d(conv3, 1, [3, 3], rate=1, activation_fn=lrelu, scope='en_conv_4')
#
#         L_enhance = tf.sigmoid(conv4)
#     return L_enhance

# def Illumination_adjust_net(input_i):
#     with tf.variable_scope('Illumination_adjust_net', reuse=tf.AUTO_REUSE):
#
#         conv1 = slim.conv2d(input_i, 32, [3, 3], rate=1, activation_fn=lrelu, scope='en_conv_1')
#         conv2 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='en_conv_2')
#         conv3 = slim.conv2d(conv2, 32, [3, 3], rate=1, activation_fn=lrelu, scope='en_conv_3')
#         conv4 = slim.conv2d(conv3, 1, [3, 3], rate=1, activation_fn=lrelu, scope='en_conv_4')
#
#         L_enhance = tf.sigmoid(conv4)
#     return L_enhance

def SKFF(input1, input2, height, n_filters, ratio, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        channel = input1.get_shape()[-1]
        batch_size = input1.get_shape()[0]
        inp_feats = tf.concat([input1, input2], axis=3)
        # [N, height, H, W, C]
        inp_feats = inp_feats.view(batch_size, height, input1.get_shape()[2], input1.get_shape()[3], channel)
        # L [N, H, W, C]
        feats_U = tf.reduce_sum(inp_feats, axis=1)
        # S = GAP(L) [N, 1, 1, C]
        feats_S = tf.reduce_mean(feats_U, axis=[1, 2], keepdims=True)
        # Z=LRelu(Conv(S))
        d = max(int(channel / ratio), 4)
        feats_Z = tf.layers.conv2d(feats_S, filter=d, kernel_size=[1, 1], strides=[1, 1], padding="same",
                                          activation=lrelu, name='SKFF_conv0')
        # V1=Conv(Z),V2=Conv(Z)
        attention_vectors1 = tf.layers.conv2d(feats_Z, filter=channel, kernel_size=[1, 1], strides=[1, 1], padding="same",
                                          activation=None, name='SKFF_conv1_1')
        attention_vectors2 = tf.layers.conv2d(feats_Z, filter=channel, kernel_size=[1, 1], strides=[1, 1],
                                              padding="same",
                                              activation=None, name='SKFF_conv1_2')
        # 转换到通道维度，再按照特征向量数量分开
        attention_vectors = tf.concat([attention_vectors1, attention_vectors2], dim=1)
        attention_vectors = attention_vectors.view(batch_size, height, 1, 1, n_filters)
        # S=softmax(V)
        attention_vectors = tf.nn.softmax(attention_vectors, axis=-1)
        # sum U=L1*S1+L2*S2
        feats_V = tf.reduce_sum(inp_feats * attention_vectors, dim=1)
        return feats_V

def ContextBlock(input, n_filters, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # modeling
        batch, height, width, channel = input.get_shape()
        # input_x = Fb
        input_x = input
        # 分支1--------------
        # [N, H * W, C]
        input_x = input_x.view(batch, height * width, channel)
        # [N, 1, H * W, C]
        input_x = np.expand_dims(input_x, axis=1)

        # 分支2--------------
        # input[N, H, W, C]-> modeling_conv1-> [N, H, W, 1]
        modeling_conv = tf.layers.conv2d(input, filter=1, kernel_size=[1, 1], strides=[1, 1], padding="same",
                                          activation=None, name='modeling_conv')
        # modeling_conv1[N, H*W, 1]
        modeling_conv = modeling_conv.view(batch, height * width, 1)
        # [N, H*W, 1] Fc, 经过softmax后维度不变
        modeling_conv = tf.nn.softmax(modeling_conv, axis=-1)
        # [N, H*W, 1, 1]
        modeling_conv = np.expand_dims(modeling_conv, axis=3)
        # input_x[N, 1, H * W, C], modeling_conv1[N, H * W, 1, 1]
        # Fd [N, 1, C, 1] 改input_x[N, 1, C, H * W], modeling_conv1[N, 1, H * W, 1]
        context = tf.matmul(input_x.view(batch, 1, channel, height * width),
                            modeling_conv.view(batch, 1, height * width, 1))
        # [N, 1, 1, C]
        context = context.view(batch, 1, 1, channel)

        # transform,[N, 1, 1, C]
        # conv+LRelu+conv->Fe,[N, 1, 1, C]
        transform_conv = tf.layers.conv2d(context, filter=n_filters, kernel_size=[1, 1], strides=[1, 1], padding="same",
                                          activation=lrelu, name='transform_conv')
        transform_conv1 = tf.layers.conv2d(transform_conv, filter=n_filters, kernel_size=[1, 1], strides=[1, 1], padding="same",
                                          activation=lrelu, name='transform_conv1')
        # Fusion
        input = input + transform_conv1
        input = lrelu(input)
        return input

def RCB(input, n_filters, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # Fa->Conv+LRelu+Conv->Fb
        RCB_conv1 = tf.layers.conv2d(input, filters=n_filters, kernel_size=[3, 3], strides=[1, 1], padding = "same", activation = lrelu, name='RCB_conv1')
        RCB_conv2 = tf.layers.conv2d(RCB_conv1, filters=n_filters, kernel_size=[3, 3], strides=[1, 1], padding="same", activation=None, name='RCB_conv2')
        # Modeling+Transform+Fusion+LRelu
        RBC_ContextBlock = ContextBlock(RCB_conv2, n_filters, name='ContextBlock')
        # Res
        res = RBC_ContextBlock + input
        return res

def RRG(input, n_filters, name):
    with tf.variable_scope(name):
        # Res
        res = res + input
        return res

def MIRNet(input_r, input_i):
    with tf.variable_scope('MIRNet', reuse=tf.AUTO_REUSE):
        # 给反射图像引入attention
        conv0_0 = slim.conv2d(input_r, 32, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv0_0')
        attention0 = gam_block(conv0_0, name='attention_0', ratio=4)
        # 亮度图像引入attention
        conv0_1 = slim.conv2d(input_i, 4, [3, 3], rate=1, activation_fn=tf.nn.sigmoid, scope='conv0_1')
        attention = spatial_attention1(conv0_1, name='attention', ratio=4)
        # 合并
        input_all = tf.concat([attention0, attention], 3)
        # MIRNet,channel
        n_filters = 80
        MIRNet_conv1 = tf.layers.conv2d(input, filters=n_filters, kernel_size = [3, 3], strides = [1, 1], padding = "same", activation = None, name='MIRNet_conv1')
        rrg1 = RRG(MIRNet_conv1, name='rrg1')
        rrg2 = RRG(rrg1, n_filters, name='rrg2')
        rrg3 = RRG(rrg2, n_filters, name='rrg3')
        rrg4 = RRG(rrg3, n_filters, name='rrg4')
        # R
        MIRNet_conv2 = tf.layers.conv2dtf.layers.conv2d(rrg4, filters=3, kernel_size = [3, 3], strides = [1, 1], padding = "same", activation = None, name='MIRNet_conv2')
        # I'
        out_img = MIRNet_conv2 + input_all
        return out_img