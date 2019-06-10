from __future__ import print_function
from math import ceil

import tensorflow as tf

from model_util.pspnet.layers_utils import Conv2D, MaxPooling2D, AveragePooling2D
from model_util.pspnet.layers_utils import BN, Activation, Dropout, ZeroPadding2D_symmetric, GroupNorm
from model_util.pspnet.layers_utils import Concatenate, Add

import cv2
import numpy as np



def Interp(x, shape, method=tf.image.ResizeMethod.BILINEAR):
    new_height, new_width = shape
    resized = tf.image.resize_images(x, [new_height, new_width],
                                     align_corners=True,
                                     method=method)
    return resized


def residual_conv(prev, level, pad=1, lvl=1, sub_lvl=1, modify_stride=False, is_training=True, use_gn=False):
    lvl = str(lvl)
    sub_lvl = str(sub_lvl)
    names = ["conv"+lvl+"_" + sub_lvl + "_1x1_reduce",
             "conv"+lvl+"_" + sub_lvl + "_1x1_reduce_bn",
             "conv"+lvl+"_" + sub_lvl + "_3x3",
             "conv"+lvl+"_" + sub_lvl + "_3x3_bn",
             "conv"+lvl+"_" + sub_lvl + "_1x1_increase",
             "conv"+lvl+"_" + sub_lvl + "_1x1_increase_bn"]
    if modify_stride is False:
        prev = Conv2D(prev, 64 * level, (1, 1), strides=(1, 1), name=names[0],
                      use_bias=False)
    elif modify_stride is True:
        prev = Conv2D(prev, 64 * level, (1, 1), strides=(2, 2), name=names[0],
                      use_bias=False)

    prev = GroupNorm(prev, 4 * level, name=names[0]+'_gn') if use_gn else BN(prev, name=names[1], is_training=is_training)
    prev = Activation(prev, 'relu')

    prev = ZeroPadding2D_symmetric(prev, padding=pad)
    prev = Conv2D(prev, 64 * level, (3, 3), strides=(1, 1), dilation_rate=pad,
                  name=names[2], use_bias=False)

    prev = GroupNorm(prev, 4 * level, name=names[2]+'_gn') if use_gn else BN(prev, name=names[3], is_training=is_training)
    prev = Activation(prev, 'relu')
    prev = Conv2D(prev, 256 * level, (1, 1), strides=(1, 1), name=names[4],
                  use_bias=False)
    prev = GroupNorm(prev, 16 * level, name=names[4]+'_gn') if use_gn else BN(prev, name=names[5], is_training=is_training)
    return prev


def short_convolution_branch(prev, level, lvl=1, sub_lvl=1, modify_stride=False, is_training=True, use_gn=False):
    lvl = str(lvl)
    sub_lvl = str(sub_lvl)
    names = ["conv" + lvl+"_" + sub_lvl + "_1x1_proj",
             "conv" + lvl+"_" + sub_lvl + "_1x1_proj_bn"]

    if modify_stride is False:
        prev = Conv2D(prev, 256 * level, (1, 1), strides=(1, 1), name=names[0],
                      use_bias=False)
    elif modify_stride is True:
        prev = Conv2D(prev, 256 * level, (1, 1), strides=(2, 2), name=names[0],
                      use_bias=False)

    prev = GroupNorm(prev, 16 * level, name=names[0]+'_gn') if use_gn else BN(prev, name=names[1], is_training=is_training)
    return prev


def empty_branch(prev):
    return prev


def residual_short(prev_layer, level, pad=1, lvl=1, sub_lvl=1, modify_stride=False, is_training=True, use_gn=False):
    prev_layer = Activation(prev_layer, 'relu')
    block_1 = residual_conv(prev_layer, level,
                            pad=pad, lvl=lvl, sub_lvl=sub_lvl,
                            modify_stride=modify_stride,
                            is_training=is_training,
                            use_gn=use_gn)

    block_2 = short_convolution_branch(prev_layer, level,
                                       lvl=lvl, sub_lvl=sub_lvl,
                                       modify_stride=modify_stride,
                                       is_training=is_training,
                                       use_gn=use_gn)
    added = Add([block_1, block_2])
    return added


def residual_empty(prev_layer, level, pad=1, lvl=1, sub_lvl=1, is_training=True, use_gn=False):
    prev_layer = Activation(prev_layer, 'relu')

    block_1 = residual_conv(prev_layer, level, pad=pad,
                            lvl=lvl, sub_lvl=sub_lvl, is_training=is_training, use_gn=use_gn)
    block_2 = empty_branch(prev_layer)
    added = Add([block_1, block_2])
    return added


def ResNet(inp, layers, is_training=True):
    # Names for the first couple layers of model_util
    names = ["conv1_1_3x3_s2",
             "conv1_1_3x3_s2_bn",
             "conv1_2_3x3",
             "conv1_2_3x3_bn",
             "conv1_3_3x3",
             "conv1_3_3x3_bn"]

    # Short branch(only start of network)

    cnv1 = Conv2D(inp, 64, (3, 3), strides=(2, 2), padding='same', name=names[0],
                  use_bias=False)  # "conv1_1_3x3_s2"
    bn1 = BN(cnv1, name=names[1], is_training=is_training)  # "conv1_1_3x3_s2/bn"
    relu1 = Activation(bn1, 'relu')  # "conv1_1_3x3_s2/relu"

    cnv1 = Conv2D(relu1, 64, (3, 3), strides=(1, 1), padding='same', name=names[2],
                  use_bias=False)  # "conv1_2_3x3"
    bn1 = BN(cnv1, name=names[3], is_training=is_training)  # "conv1_2_3x3/bn"
    relu1 = Activation(bn1, 'relu')  # "conv1_2_3x3/relu"

    cnv1 = Conv2D(relu1, 128, (3, 3), strides=(1, 1), padding='same', name=names[4],
                  use_bias=False)  # "conv1_3_3x3"
    bn1 = BN(cnv1, name=names[5], is_training=is_training)  # "conv1_3_3x3/bn"
    relu1 = Activation(bn1, 'relu')  # "conv1_3_3x3/relu"

    res = MaxPooling2D(relu1, pool_size=(3, 3), padding='same',
                       strides=(2, 2))  # "pool1_3x3_s2"

    # ---Residual layers(body of network)

    """
    Modify_stride --Used only once in first 3_1 convolutions block.
    changes stride of first convolution from 1 -> 2
    """

    # 2_1- 2_3
    res = residual_short(res, 1, pad=1, lvl=2, sub_lvl=1, is_training=is_training)
    for i in range(2):
        res = residual_empty(res, 1, pad=1, lvl=2, sub_lvl=i+2, is_training=is_training)

    # 3_1 - 3_3
    res = residual_short(res, 2, pad=1, lvl=3, sub_lvl=1, modify_stride=True, is_training=is_training)
    for i in range(3):
        res = residual_empty(res, 2, pad=1, lvl=3, sub_lvl=i+2, is_training=is_training)
    if layers is 50:
        # 4_1 - 4_6
        res = residual_short(res, 4, pad=2, lvl=4, sub_lvl=1, is_training=is_training)
        for i in range(5):
            res = residual_empty(res, 4, pad=2, lvl=4, sub_lvl=i+2, is_training=is_training)
    elif layers is 101:
        # 4_1 - 4_23
        res = residual_short(res, 4, pad=2, lvl=4, sub_lvl=1, is_training=is_training)
        for i in range(22):
            res = residual_empty(res, 4, pad=2, lvl=4, sub_lvl=i+2, is_training=is_training)
    else:
        print("This ResNet is not implemented")

    # 5_1 - 5_3
    res = residual_short(res, 8, pad=4, lvl=5, sub_lvl=1, is_training=is_training)
    for i in range(2):
        res = residual_empty(res, 8, pad=4, lvl=5, sub_lvl=i+2, is_training=is_training)

    res = Activation(res, 'relu')
    return res


def ResNet_stem(inputs, is_training=True):
    # Names for the first couple layers of model_util
    names = ["conv1_1_3x3_s2",
             "conv1_1_3x3_s2_bn",
             "conv1_2_3x3",
             "conv1_2_3x3_bn",
             "conv1_3_3x3",
             "conv1_3_3x3_bn"]

    # Short branch(only start of network)
    inp = inputs

    cnv1 = Conv2D(inp, 64, (3, 3), strides=(2, 2), padding='same', name=names[0],
                  use_bias=False)  # "conv1_1_3x3_s2"
    bn1 = BN(cnv1, name=names[1], is_training=is_training)  # "conv1_1_3x3_s2/bn"
    relu1 = Activation(bn1, 'relu')  # "conv1_1_3x3_s2/relu"

    cnv1 = Conv2D(relu1, 64, (3, 3), strides=(1, 1), padding='same', name=names[2],
                  use_bias=False)  # "conv1_2_3x3"
    bn1 = BN(cnv1, name=names[3], is_training=is_training)  # "conv1_2_3x3/bn"
    relu1 = Activation(bn1, 'relu')  # "conv1_2_3x3/relu"

    cnv1 = Conv2D(relu1, 128, (3, 3), strides=(1, 1), padding='same', name=names[4],
                  use_bias=False)  # "conv1_3_3x3"
    bn1 = BN(cnv1, name=names[5], is_training=is_training)  # "conv1_3_3x3/bn"
    relu1 = Activation(bn1, 'relu')  # "conv1_3_3x3/relu"

    res = MaxPooling2D(relu1, pool_size=(3, 3), padding='same',
                       strides=(2, 2))  # "pool1_3x3_s2"

    # ---Residual layers(body of network)

    """
    Modify_stride --Used only once in first 3_1 convolutions block.
    changes stride of first convolution from 1 -> 2
    """

    # 2_1- 2_3
    res = residual_short(res, 1, pad=1, lvl=2, sub_lvl=1, is_training=is_training)
    for i in range(2):
        res = residual_empty(res, 1, pad=1, lvl=2, sub_lvl=i+2, is_training=is_training)

    # 3_1 - 3_3
    res = residual_short(res, 2, pad=1, lvl=3, sub_lvl=1, modify_stride=True, is_training=is_training)
    for i in range(3):
        res = residual_empty(res, 2, pad=1, lvl=3, sub_lvl=i+2, is_training=is_training)

    # 4_1 - 4_6
    res = residual_short(res, 4, pad=2, lvl=4, sub_lvl=1, is_training=is_training)
    for i in range(5):
        res = residual_empty(res, 4, pad=2, lvl=4, sub_lvl=i + 2, is_training=is_training)

    return res


def interp_block(prev_layer, level, feature_map_shape, str_lvl=1, dims=None, name=None, is_training=True, use_gn=False):
    if name :
        names = [
            name + "_conv",
            name + "_conv_bn"
        ]
    else:
        str_lvl = str(str_lvl)

        names = [
            "conv5_3_pool"+str_lvl+"_conv",
            "conv5_3_pool"+str_lvl+"_conv_bn"
            ]

    out_dims = dims if dims else 512

    kernel = (int(ceil(feature_map_shape[0]/level)), int(ceil(feature_map_shape[1]/level)))
    strides = (int(ceil(feature_map_shape[0]/level)), int(ceil(feature_map_shape[1]/level)))
    prev_layer = AveragePooling2D(prev_layer, kernel, strides=strides)
    prev_layer = Conv2D(prev_layer, out_dims, (1, 1), strides=(1, 1), name=names[0],
                        use_bias=False)
    prev_layer = GroupNorm(prev_layer, out_dims/16, name=names[0]+'_gn') if use_gn else BN(prev_layer, name=names[1], is_training=is_training)
    prev_layer = Activation(prev_layer, 'relu')
    prev_layer = Interp(prev_layer, shape=feature_map_shape)

    return prev_layer


def build_pyramid_pooling_module(res, is_training=True, use_gn=False):
    """Build the Pyramid Pooling Module."""
    # ---PSPNet concat layers with Interpolation
    #feature_map_size = tuple(int(ceil(input_dim / 8.0)) for input_dim in input_shape)
    feature_map_size = res.shape.as_list()[1:3]
    print("PSP module will interpolate to a final feature map size of %s" % (feature_map_size, ))

    interp_block1 = interp_block(res, 1, feature_map_size, str_lvl=1, is_training=is_training, use_gn=use_gn)
    interp_block2 = interp_block(res, 2, feature_map_size, str_lvl=2, is_training=is_training, use_gn=use_gn)
    interp_block3 = interp_block(res, 3, feature_map_size, str_lvl=3, is_training=is_training, use_gn=use_gn)
    interp_block6 = interp_block(res, 6, feature_map_size, str_lvl=6, is_training=is_training, use_gn=use_gn)

    # concat all these layers. resulted shape=(1,feature_map_size_x,feature_map_size_y,4096)
    res = Concatenate([res,
                       interp_block6,
                       interp_block3,
                       interp_block2,
                       interp_block1])
    return res


def build_pspnet(resnet_layers,
                 inputs,
                 nb_classes,
                 dynamic_envs,
                 activation='softmax',
                 is_training=True):
    dropout_ratio = 0.1 if is_training else 0

    inp = tf.concat([inputs, dynamic_envs], axis=-1)
    inputs_shape_spatial = inp.shape.as_list()[1:-1]
    res = ResNet(inp, layers=resnet_layers, is_training=is_training)
    psp = build_pyramid_pooling_module(res, is_training=is_training)

    '''
    # Prior
    x_prior = Conv2D(psp, 512, (3, 3), strides=(1, 1), padding="same", name="conv5_4",
               use_bias=False)
    x_prior = BN(x_prior, name="conv5_4_bn", is_training=is_training)
    x_prior = Activation(x_prior, 'relu')
    x_prior = Dropout(x_prior, dropout_ratio)
    x_prior = Conv2D(x_prior, nb_classes, (1, 1), strides=(1, 1), padding="same", name="conv6")
    prior = Activation(x_prior, activation)
    '''

    # Top
    with tf.variable_scope('top') as _:

        x = Conv2D(psp, 1024, (3, 3), strides=(1, 1), padding="same", name="conv_top_1",
                   use_bias=False)
        x = BN(x, name="conv_top_1_bn", is_training=is_training)
        x = Activation(x, 'relu')

        top_shortcut = x

        x = Conv2D(x, 1024, (3, 3), strides=(1, 1), padding="same", name="conv_top_2",
                   use_bias=False)
        x = BN(x, name="conv_top_2_bn", is_training=is_training)
        x = Activation(x, 'relu')


        x = Conv2D(x, 1024, (3, 3), strides=(1, 1), padding="same", name="conv_top_3",
                   use_bias=False)
        x = BN(x, name="conv_top_3_bn", is_training=is_training)
        x = Activation(x, 'relu')

        x = x + top_shortcut

        x = Dropout(x, dropout_ratio)

        preact_policy = Conv2D(x, nb_classes, (1, 1), strides=(1, 1), padding="same", name="conv_class")
        interp_preact_policy = Interp(x=preact_policy,
                                      shape=inputs_shape_spatial)

        state_value = Conv2D(x, 1, (1, 1), strides=(1, 1), name="conv_value", use_bias=False)

        policy = Activation(preact_policy, 'softmax')
        interp_policy = Activation(interp_preact_policy, 'softmax')

    return tf.squeeze(state_value, axis=-1), (preact_policy, policy, interp_preact_policy, interp_policy)


def build_pspnet_prior(nb_classes, inputs, activation='softmax', is_training=True):
    """Build PSPNet."""
    dropout_ratio = 0.1 if is_training else 0

    x = Conv2D(inputs, 512, (3, 3), strides=(1, 1), padding="same", name="conv5_4",
               use_bias=False)
    x = BN(x, name="conv5_4_bn", is_training=is_training)
    x = Activation(x, 'relu')
    x = Dropout(x, dropout_ratio)

    x = Conv2D(x, nb_classes, (1, 1), strides=(1, 1), padding="same", name="conv6")
    prior = Activation(x, activation)

    return Interp(prior, [713, 713])


def build_pspnet_ade20k_prior(nb_classes, inputs, activation='softmax', is_training=True):
    """Build PSPNet."""
    dropout_ratio = 0.1 if is_training else 0

    x_dd = Conv2D(inputs, 512, (3, 3), strides=(1, 1), padding="same", name="conv5_4",
               use_bias=False)
    x_dd = BN(x_dd, name="conv5_4_bn", is_training=is_training)
    x_dd = Activation(x_dd, 'relu')


    x_dd = Dropout(x_dd, dropout_ratio)

    prior = Conv2D(x_dd, nb_classes, (1, 1), strides=(1, 1), padding="same", name="conv6")
    prior = Activation(prior, activation)

    x_nd = Conv2D(inputs, 512, (3, 3), strides=(1, 1), padding="same", name="base_prediction_conv5_4",
                  use_bias=False)
    x_nd = GroupNorm(x_nd, 32, name="base_prediction_conv5_4_gn")
    x_nd = Activation(x_nd, 'relu')
    x_nd = Dropout(x_nd, dropout_ratio)

    base_prediction = Conv2D(x_nd, nb_classes, (1, 1), strides=(1, 1), padding="same", name="base_prediction_conv6")

    return base_prediction, Interp(prior, [473, 473])


def build_pspnet50_prior_on_resnet_stem(nb_classes, inputs, activation='softmax', is_training=True):
    """Build PSPNet."""
    dropout_ratio = 0.1 if is_training else 0

    inp = inputs

    res = residual_short(inp, 8, pad=4, lvl=5, sub_lvl=1, is_training=is_training)
    for i in range(2):
        res = residual_empty(res, 8, pad=4, lvl=5, sub_lvl=i + 2, is_training=is_training)

    res = Activation(res, 'relu')

    psp = build_pyramid_pooling_module(res, is_training=is_training)

    x = Conv2D(psp, 512, (3, 3), strides=(1, 1), padding="same", name="conv5_4",
               use_bias=False)
    x = BN(x, name="conv5_4_bn", is_training=is_training)
    x = Activation(x, 'relu')
    x = Dropout(x, dropout_ratio)

    x = Conv2D(x, nb_classes, (1, 1), strides=(1, 1), padding="same", name="conv6")
    prior = Activation(x, activation)

    return Interp(prior, [473, 473])


def build_pspnet_basic(resnet_layers, inputs, is_training=True):
    """Build PSPNet."""
    inp = inputs
    res = ResNet(inp, layers=resnet_layers, is_training=is_training)
    psp = build_pyramid_pooling_module(res, is_training=is_training)

    return psp


def build_pspnet_basic_101(inputs, is_training=True):
    """Build PSPNet."""
    inp = inputs
    res = ResNet(inp, layers=101, is_training=is_training)
    psp = build_pyramid_pooling_module(res, is_training=is_training)

    return psp


def build_pspnet_basic_50(inputs, is_training=True):
    """Build PSPNet."""
    inp = inputs
    res = ResNet(inp, layers=50, is_training=is_training)
    psp = build_pyramid_pooling_module(res, is_training=is_training)

    return psp


def pspnet_top_cls(inputs, knowledges, base_prediction, nb_classes, activation='softmax', is_knowledge_empty=False, is_training=True):
    dynamic_envs_size = knowledges.shape.as_list()[1:3]
    dropout_ratio = 0.1 if is_training else 0
    with tf.variable_scope('top_cls') as _:
        #dim 64 -> dim 256
        knowledges_1 = interp_block(knowledges, 1, dynamic_envs_size, name='knowledges_proj_1', dims=256, use_gn=True, is_training=is_training)
        knowledges_2 = interp_block(knowledges, 2, dynamic_envs_size, name='knowledges_proj_2', dims=256, use_gn=True, is_training=is_training)
        knowledges_3 = interp_block(knowledges, 3, dynamic_envs_size, name='knowledges_proj_3', dims=256, use_gn=True, is_training=is_training)
        knowledges_6 = interp_block(knowledges, 6, dynamic_envs_size, name='knowledges_proj_6', dims=256, use_gn=True, is_training=is_training)
        knowledges_x = Concatenate([inputs, knowledges, knowledges_6, knowledges_3,knowledges_2, knowledges_1])
        '''
        envs_z = Concatenate([dynamic_envs, envs_6, envs_3, envs_2, envs_1])

        z = Conv2D(envs_z, 256, (3, 3), strides=(1, 1), padding="same", name="conv_top_cls_dyenv_1",
                   use_bias=False)
        #z = BN(z, name="conv_top_cls_1_1_bn", is_training=is_training)
        z = GroupNorm(z, 16, name='conv_top_cls_dyenv_1_gn')
        z = Activation(z, 'relu')

        top_shortcut_z = z

        z = Conv2D(z, 256, (3, 3), strides=(1, 1), padding="same", name="conv_top_cls_dyenv_2",
                   use_bias=False)
        #z = BN(z, name="conv_top_cls_1_2_bn", is_training=is_training)
        z = GroupNorm(z, 16, name='conv_top_cls_dyenv_2_gn')
        z = Activation(z, 'relu')

        z = Conv2D(z, 256, (3, 3), strides=(1, 1), padding="same", name="conv_top_cls_dyenv_3",
                   use_bias=False)
        #z = BN(z, name="conv_top_cls_1_3_bn", is_training=is_training)
        z = GroupNorm(z, 16, name='conv_top_cls_dyenv_3_gn')
        z = Activation(z, 'relu')

        z = z + top_shortcut_z
        '''
        x = Conv2D(knowledges_x, 512, (3, 3), strides=(1, 1), padding="same", name="conv_top_cls_env_1",
                   use_bias=False)
        #x = BN(x, name="conv_top_cls_1_bn", is_training=is_training)
        x = GroupNorm(x, 32, name='conv_top_cls_knowledges_1_gn')
        x = Activation(x, 'relu')

        top_shortcut_x = x

        x = Conv2D(x, 512, (3, 3), strides=(1, 1), padding="same", name="conv_top_cls_knowledges_2",
                   use_bias=False)
        #x = BN(x, name="conv_top_cls_2_bn", is_training=is_training)
        x = GroupNorm(x, 32, name='conv_top_cls_knowledges_2_gn')
        x = Activation(x, 'relu')


        x = Conv2D(x, 512, (3, 3), strides=(1, 1), padding="same", name="conv_top_cls_knowledges_3",
                   use_bias=False)
        #x = BN(x, name="conv_top_cls_3_bn", is_training=is_training)
        x = GroupNorm(x, 32, name='conv_top_cls_knowledges_3_gn')
        x = Activation(x, 'relu')

        x = x + top_shortcut_x

        #zx = Concatenate([z, x])
        zx = x
        #top_shortcut_zx = zx

        zx = Conv2D(zx, 512, (3, 3), strides=(1, 1), padding="same", name="conv_top_cls_knowledges",
                    use_bias=False)
        zx = GroupNorm(zx, 32, name='conv_top_cls_knowledges_gn')
        zx = Activation(zx, 'relu')

        zx = zx + top_shortcut_x

        zx = Dropout(zx, dropout_ratio)

        zx = Conv2D(zx, nb_classes, (1, 1), strides=(1, 1), padding="same", use_bias=False, name="conv_class")

        #x_activation = Activation(x, activation)
        if not is_knowledge_empty:
            return zx + base_prediction
        else:
            return base_prediction


def pspnet_top_cls_v2(inputs, dynamic_envs, nb_classes, activation='softmax', is_training=True):
    dropout_ratio = 0.1 if is_training else 0
    with tf.variable_scope('top_cls_v2') as _:
        feature_with_envs = Concatenate([inputs, dynamic_envs])

        res = residual_short(feature_with_envs, 8, pad=4, lvl=5, sub_lvl=1, is_training=is_training, use_gn=True)
        for i in range(2):
            res = residual_empty(res, 8, pad=4, lvl=5, sub_lvl=i + 2, is_training=is_training, use_gn=True)

        res = Activation(res, 'relu')

        psp = build_pyramid_pooling_module(res, is_training=is_training, use_gn=True)

        x = Conv2D(psp, 512, (3, 3), strides=(1, 1), padding="same", name="conv5_4",
                   use_bias=False)
        x = GroupNorm(x, 32, name="conv5_4_gn")
        x = Activation(x, 'relu')
        x = Dropout(x, dropout_ratio)

        x = Conv2D(x, nb_classes, (1, 1), strides=(1, 1), padding="same", name="conv6")
        x = Activation(x, activation)

        return x


def pspnet_top_plc(inputs, dynamic_envs, inputs_shape_spatial, is_training=True):
    dynamic_envs_size = dynamic_envs.shape.as_list()[1:3]
    dropout_ratio = 0.1 if is_training else 0
    with tf.variable_scope('top_plc') as _:
        envs_1 = interp_block(dynamic_envs, 1, dynamic_envs_size, name='env_proj_1', dims=64, is_training=is_training)
        envs_2 = interp_block(dynamic_envs, 2, dynamic_envs_size, name='env_proj_2', dims=64, is_training=is_training)
        envs_3 = interp_block(dynamic_envs, 3, dynamic_envs_size, name='env_proj_3', dims=64, is_training=is_training)
        envs_6 = interp_block(dynamic_envs, 6, dynamic_envs_size, name='env_proj_6', dims=64, is_training=is_training)
        envs_x = Concatenate([inputs, dynamic_envs, envs_6, envs_3, envs_2, envs_1])
        '''
        envs_z = Concatenate([dynamic_envs, envs_6, envs_3, envs_2, envs_1])
        
        z = Conv2D(envs_z, 256, (3, 3), strides=(1, 1), padding="same", name="conv_top_plc_dyenv_1",
                   use_bias=False)
        # z = BN(z, name="conv_top_cls_1_1_bn", is_training=is_training)
        z = GroupNorm(z, 16, name='conv_top_plc_dyenv_1_gn')
        z = Activation(z, 'relu')

        top_shortcut_z = z

        z = Conv2D(z, 256, (3, 3), strides=(1, 1), padding="same", name="conv_top_plc_dyenv_2",
                   use_bias=False)
        # z = BN(z, name="conv_top_cls_1_2_bn", is_training=is_training)
        z = GroupNorm(z, 16, name='conv_top_plc_dyenv_2_gn')
        z = Activation(z, 'relu')

        z = Conv2D(z, 256, (3, 3), strides=(1, 1), padding="same", name="conv_top_plc_dyenv_3",
                   use_bias=False)
        # z = BN(z, name="conv_top_cls_1_3_bn", is_training=is_training)
        z = GroupNorm(z, 16, name='conv_top_plc_dyenv_3_gn')
        z = Activation(z, 'relu')

        z = z + top_shortcut_z
        '''
        x = Conv2D(envs_x, 512, (3, 3), strides=(1, 1), padding="same", name="conv_top_plc_env_1",
                   use_bias=False)
        # x = BN(x, name="conv_top_cls_1_bn", is_training=is_training)
        x = GroupNorm(x, 32, name='conv_top_plc_env_1_gn')
        x = Activation(x, 'relu')

        top_shortcut_x = x

        x = Conv2D(x, 512, (3, 3), strides=(1, 1), padding="same", name="conv_top_plc_env_2",
                   use_bias=False)
        # x = BN(x, name="conv_top_cls_2_bn", is_training=is_training)
        x = GroupNorm(x, 32, name='conv_top_plc_env_2_gn')
        x = Activation(x, 'relu')

        x = Conv2D(x, 512, (3, 3), strides=(1, 1), padding="same", name="conv_top_plc_env_3",
                   use_bias=False)
        # x = BN(x, name="conv_top_cls_3_bn", is_training=is_training)
        x = GroupNorm(x, 32, name='conv_top_plc_env_3_gn')
        x = Activation(x, 'relu')

        x = x + top_shortcut_x

        #zx = Concatenate([z, x])
        zx = x
        # top_shortcut_zx = zx

        zx = Conv2D(zx, 512, (3, 3), strides=(1, 1), padding="same", name="conv_top_plc_dyenvs",
                    use_bias=False)
        zx = GroupNorm(zx, 32, name='conv_top_plc_dyenvs_gn')
        zx = Activation(zx, 'relu')

        zx = zx + top_shortcut_x

        zx = Dropout(zx, dropout_ratio)

        preact_policy = Conv2D(zx, 2, (1, 1), strides=(1, 1), padding="same", name="conv_policy")
        interp_preact_policy = Interp(x=preact_policy,
                                      shape=inputs_shape_spatial,
                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        state_value = Conv2D(zx, 1, (1, 1), strides=(1, 1), name="conv_value", use_bias=False)

        policy = Activation(preact_policy, 'softmax')
        interp_policy = Activation(interp_preact_policy, 'softmax')

        return tf.squeeze(state_value, axis=-1), (preact_policy, policy, interp_preact_policy, interp_policy)


def pspnet_top_plc_v2(inputs, dynamic_envs, inputs_shape_spatial, is_training=True):
    dropout_ratio = 0.1 if is_training else 0
    with tf.variable_scope('top_plc_v2') as _:
        feature_with_envs = Concatenate([inputs, dynamic_envs])

        res = residual_short(feature_with_envs, 8, pad=4, lvl=5, sub_lvl=1, is_training=is_training, use_gn=True)
        for i in range(2):
            res = residual_empty(res, 8, pad=4, lvl=5, sub_lvl=i + 2, is_training=is_training, use_gn=True)

        res = Activation(res, 'relu')

        psp = build_pyramid_pooling_module(res, is_training=is_training, use_gn=True)

        x = Conv2D(psp, 512, (3, 3), strides=(1, 1), padding="same", name="conv5_4",
                   use_bias=False)
        x = GroupNorm(x, 32, name="conv5_4_gn")
        x = Activation(x, 'relu')
        x = Dropout(x, dropout_ratio)

        preact_policy = Conv2D(x, 2, (1, 1), strides=(1, 1), padding="same", name="conv_policy")
        interp_preact_policy = Interp(x=preact_policy,
                                      shape=inputs_shape_spatial,
                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        state_value = Conv2D(x, 1, (1, 1), strides=(1, 1), name="conv_value", use_bias=False)

        policy = Activation(preact_policy, 'softmax')
        interp_policy = Activation(interp_preact_policy, 'softmax')

        return tf.squeeze(state_value, axis=-1), (preact_policy, policy, interp_preact_policy, interp_policy)




if __name__ == '__main__':
    iplhd = tf.placeholder(dtype=tf.float32, shape=[1, 713, 713, 3])
    logits = build_pspnet(nb_classes=150,
                         resnet_layers=101,
                         inputs=iplhd)
    init_op = tf.global_variables_initializer()
    cap = cv2.VideoCapture('example_images/ade20k.jpg')
    ret, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (713, 713))
    img = np.zeros([1,713,713,3], dtype=np.float32)

    with tf.Session() as sess:
        _ = sess.run(init_op)
        lv = sess.run(logits,
                          feed_dict={
                              iplhd:img
                          })
    v_list = tf.global_variables()
    bias_lst = [v for v in v_list if 'bias' in v.name]
    print('true')
