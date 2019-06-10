from __future__ import print_function

import tensorflow as tf


def _variable_on_cpu(name, shape, initializer=tf.contrib.layers.xavier_initializer()):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
  return var


def BN(inputs, is_training=True, name=""):

    return tf.layers.batch_normalization(inputs = inputs,
                                         momentum=0.95,
                                         epsilon=1e-5,
                                         training=is_training,
                                         name=name)


def GroupNorm(inputs, G, eps=1e-5, name=""):

    inputs = tf.transpose(inputs, [0, 3, 1, 2])
    _N, _C, _H, _W = inputs.shape.as_list()
    inputs = tf.reshape(inputs, [_N, int(G), _C//int(G), _H, _W])

    _mean, _var = tf.nn.moments(inputs, [2, 3, 4], keep_dims=True)
    inputs = (inputs - _mean) / tf.sqrt(_var + eps)

    inputs = tf.reshape(inputs, [_N, _C, _H, _W])

    gamma = _variable_on_cpu(name + '/GN_gamma',
                             shape=[1, _C, 1, 1],
                             initializer=tf.constant_initializer(1.0))
    beta = _variable_on_cpu(name + '/GN_beta',
                            shape=[1, _C, 1, 1],
                            initializer=tf.constant_initializer(0.0))

    return tf.transpose(inputs * gamma + beta, [0, 2, 3, 1])


def Conv2D(inputs,
           filters,
           kernel_size,
           strides=(1,1),
           padding='valid',
           dilation_rate=(1,1),
           use_bias=True,
           name=''):
    _in_channels = inputs.shape.as_list()[-1]
    _out_channels = filters
    padding = 'SAME' if padding == 'same' else 'VALID'

    kernel = _variable_on_cpu(name + '/kernel',
                              shape=[kernel_size[0], kernel_size[1], _in_channels, _out_channels])

    if dilation_rate == (1,1):
        conv = tf.nn.conv2d(input=inputs,
                            filter=kernel,
                            strides=[1, strides[0], strides[1], 1],
                            padding=padding)
    else:
        conv = tf.nn.atrous_conv2d(value=inputs,
                                   filters=kernel,
                                   rate=dilation_rate,
                                   padding=padding)

    if use_bias:
        biases = _variable_on_cpu(name+'/bias', _out_channels, tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
    else:
        pre_activation = conv

    '''
    return tf.layers.conv2d(inputs=inputs,
                            filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            dilation_rate=dilation_rate,
                            use_bias=use_bias,
                            padding=padding,
                            name=name)
    '''
    return pre_activation


def MaxPooling2D(inputs, pool_size, strides, padding='valid'):
    return tf.layers.max_pooling2d(inputs=inputs,
                                   pool_size=pool_size,
                                   strides=strides,
                                   padding=padding)


def AveragePooling2D(inputs, pool_size, strides, padding='valid'):
    return tf.layers.average_pooling2d(inputs=inputs,
                                       pool_size=pool_size,
                                       strides=strides,
                                       padding=padding)


def Activation(inputs ,name):
    if name == 'relu':
        return tf.nn.relu(inputs)
    elif name == 'softmax':
        return tf.nn.softmax(inputs, -1)
    elif name == 'sigmoid':
        return tf.nn.sigmoid(inputs, -1)
    else:
        print('Warning: Not valid activation name %s' % name)
        return inputs


def Dropout(inputs, rate):
    return tf.layers.dropout(inputs, rate)


def ZeroPadding2D_symmetric(inputs, padding):
    return tf.pad(inputs, [[0,0]]+[[padding, padding], [padding, padding]]+[[0,0]], "CONSTANT")


def Concatenate(inputs):
    return tf.concat(inputs, -1)


def Add(inputs):
    assert isinstance(inputs, list), "Require inputs as a list for fucntion 'Add'."

    return inputs[0] + inputs[1]