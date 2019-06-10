#!/usr/bin/env python
"""
This module is a Keras/Tensorflow based implementation of Pyramid Scene Parsing Networks.

Original paper & code published by Hengshuang Zhao et al. (2017)
"""
from __future__ import print_function
from __future__ import division

import tensorflow as tf

import model_util.pspnet.layers_builder as layers

from reinforce_utils.networker import Networker

__author__ = "Yizhou Zhou"


class PSPNetworker(Networker):
    """Pyramid Scene Parsing Network by Hengshuang Zhao et al 2017."""

    def __init__(self,
                 nb_classes,
                 inputs,
                 ground_truth,
                 weights_path=None,
                 is_training=True):
        """Instanciate a PSPNet."""
        super(PSPNetworker, self).__init__(
            nb_classes,
            inputs,
            ground_truth,
            weights_path,
            is_training,
            feature_net=layers.build_pspnet_basic_101,
            prior_net=layers.build_pspnet_prior
        )


    def _get_stval_plc(self):
        state_value, _policys = layers.pspnet_top_plc(inputs=self.feed_inputs,
                                                      dynamic_envs=self.dynamic_envs,
                                                      inputs_shape_spatial=self.original_inputs_shape_spatial,
                                                      is_training=self.is_training)
        return state_value, _policys


    def _get_preact_prd_logits(self):
        _preact_prd_logits = layers.pspnet_top_cls(inputs=self.feed_inputs,
                                                   dynamic_envs=self.dynamic_envs,
                                                   nb_classes=self.nb_classes,
                                                   activation='softmax',
                                                   is_training=self.is_training)
        return _preact_prd_logits


class PSPNetworker_ade20k(Networker):
    """Pyramid Scene Parsing Network by Hengshuang Zhao et al 2017."""

    def __init__(self,
                 nb_classes,
                 inputs,
                 ground_truth,
                 weights_path=None,
                 is_training=True):
        """Instanciate a PSPNet."""
        super(PSPNetworker_ade20k, self).__init__(
            nb_classes,
            inputs,
            ground_truth,
            weights_path,
            is_training,
            feature_net=layers.build_pspnet_basic_50,
            prior_net=layers.build_pspnet_ade20k_prior
        )


    def _get_stval_plc(self):
        state_value, _policys = layers.pspnet_top_plc(inputs=self.feed_inputs,
                                                      dynamic_envs=self.dynamic_envs,
                                                      inputs_shape_spatial=self.original_inputs_shape_spatial,
                                                      is_training=self.is_training)
        return state_value, _policys


    def _get_preact_prd_logits(self, is_knowledge_empty=False):
        _preact_prd_logits = layers.pspnet_top_cls(inputs=self.feed_inputs,
                                                   knowledges=self.knowledges,
                                                   base_prediction=self.base_prediction,
                                                   nb_classes=self.nb_classes,
                                                   activation='softmax',
                                                   is_knowledge_empty=is_knowledge_empty,
                                                   is_training=self.is_training)
        return _preact_prd_logits
