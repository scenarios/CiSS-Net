#!/usr/bin/env python
"""
This module is a Keras/Tensorflow based implementation of Pyramid Scene Parsing Networks.

Original paper & code published by Hengshuang Zhao et al. (2017)
"""
from __future__ import print_function
from __future__ import division
from os.path import splitext, join, isfile

import model_util.pspnet.layers_builder as layers

from reinforce_utils.helper.helpers import *



__author__ = "Yizhou Zhou"


# These are the means for the ImageNet pretrained ResNet
DATA_MEAN = np.array([[[[123.68, 116.779, 103.939]]]])  # RGB order


class PSPNet(object):
    """Pyramid Scene Parsing Network by Hengshuang Zhao et al 2017."""

    def __init__(self, nb_classes,
                 resnet_layers,
                 inputs,
                 ground_truth,
                 weights_path=None,
                 is_training=True):
        """Instanciate a PSPNet."""
        self.inputs = inputs
        self.nb_classes = nb_classes
        self.total_classes = self.nb_classes + 1
        self.resnet_layers = resnet_layers
        self.ground_truth = ground_truth
        self.is_training = is_training
        self.inputs_shape = inputs.shape.as_list()
        self.inputs_shape_spatial = self.inputs_shape[1:-1]


        self.__preprocess_image()


        self.middle_features = layers.build_pspnet_basic(resnet_layers=self.resnet_layers,
                                                         inputs=self.processed_inputs,
                                                         is_training=False)

        self.prior_logits = layers.build_pspnet_prior(nb_classes=self.nb_classes,
                                                      inputs=self.middle_features,
                                                      is_training=False)
        '''
        self.features_priors = tf.concat(
            (tf.stop_gradient(self.middle_features),
             tf.stop_gradient(self.prior_logits)),
            axis=-1
        )
        '''
        self.features_priors = tf.stop_gradient(self.middle_features)
        #self.features_priors = self.middle_features

        self.dynamic_envs_shape = self.features_priors.shape.as_list()[0:-1] + [self.total_classes]
        self.dynamic_envs_shape_spatial = self.dynamic_envs_shape[1:-1]

        self.dynamic_envs = tf.zeros(shape=self.dynamic_envs_shape, dtype=tf.float32)
        '''
        self.init_net = layers.build_pspnet(resnet_layers=101,
                                            inputs=self.inputs,
                                            nb_classes=self.total_classes,
                                            dynamic_envs=self.dynamic_envs,
                                            is_training=self.is_training)
        '''
        if weights_path is not None:
            self.__set_npy_weights(weights_path)

        self.envs = []
        self.actions = []
        self.rewards = []

        self.reward_dict = {0:0.92,
                            1:0.9,
                            2:0.85,
                            3:0.75,
                            4:0.5,
                            5:0.1,
                            6:-0.3,
                            7:-0.5}


    def __set_npy_weights(self, weights_path):
        """Set weights from the intermediary npy file."""
        assign_op = []
        npy_weights_path = join('C:/Users/v-yizzh/Documents/code/rl-segmentation/model_util/pspnet/weights/npy', weights_path + ".npy")
        json_path = join("weights", "keras", weights_path + ".json")
        h5_path = join("weights", "keras", weights_path + ".h5")

        print("Importing weights from %s" % npy_weights_path)
        weights = np.load(npy_weights_path, encoding="latin1").item()

        whitelist = ["InputLayer", "Activation", "ZeroPadding2D", "Add", "MaxPooling2D", "AveragePooling2D", "Lambda",
                     "Concatenate", "Dropout"]

        weights_set = 0
        for layer in tf.global_variables():
            layer_name = layer.name.split('/')[0]
            print("Processing %s //// %s" % (layer.name, layer_name))
            if layer_name[:4] == 'conv' and layer_name[-2:] == 'bn':
                if 'moving_mean' in layer.name:
                    mean = weights[layer_name]['mean'].reshape(-1)
                    assign_op.append(tf.assign(layer, mean))
                elif 'moving_variance' in layer.name:
                    variance = weights[layer_name]['variance'].reshape(-1)
                    assign_op.append(tf.assign(layer, variance))
                elif 'gamma' in layer.name:
                    scale = weights[layer_name]['scale'].reshape(-1)
                    assign_op.append(tf.assign(layer, scale))
                elif 'beta' in layer.name:
                    offset = weights[layer_name]['offset'].reshape(-1)
                    assign_op.append(tf.assign(layer, offset))
                else:
                    print("Not found weight for %s" % layer.name)
                weights_set += 1
            elif layer_name[:4] == 'conv' and not layer_name[-4:] == 'relu':
                if 'kernel' in layer.name:
                    weight = weights[layer_name.encode()]['weights'.encode()]
                    assign_op.append(tf.assign(layer, weight))
                elif 'bias' in layer.name:
                    biases = weights[layer_name]['biases']
                    assign_op.append(tf.assign(layer, biases))
                else:
                    print("Not found weight for %s" % layer.name)
                weights_set += 1
            else:
                print("Warning: Did not find weights for %s in numpy weights" % layer.name)

        print("Set a total of %i weights" % weights_set)

        print('Finished importing weights.')

        print("Writing keras model_util & weights")
        saver = tf.train.Saver()
        with tf.Session() as s:
            s.run(assign_op)
            save_path = saver.save(s, join("weights", weights_path + ".ckpt"))
            s.close()
        print("Finished writing model_util & weights to %s" % save_path)


    def __inference_once(self):
        """
        Predict segementation for an image.

        Arguments:
            img: must be rowsxcolsx3
        """
        state_value, _policys = layers.pspnet_top_plc(inputs=self.features_priors,
                                                      dynamic_envs=self.dynamic_envs,
                                                      inputs_shape_spatial=self.inputs_shape_spatial,
                                                      is_training=self.is_training)

        _preact_prd_logits = layers.pspnet_top_cls(inputs=self.features_priors,
                                           dynamic_envs=self.dynamic_envs,
                                           nb_classes=self.nb_classes,
                                           activation='softmax',
                                           is_training=self.is_training)

        _preact_policy, policy, preact_interp_policy, interp_policy = _policys

        preact_interp_prd_logits = layers.Interp(x=_preact_prd_logits,
                                                 shape=self.inputs_shape_spatial)

        action = self.__gen_action(policy)
        NS_action = tf.expand_dims(tf.argmax(policy, axis=-1), axis=-1)
        NS_action_interp_smooth = tf.expand_dims(
            tf.argmax(
                layers.Interp(x=policy,
                              shape=self.inputs_shape_spatial,
                              method=tf.image.ResizeMethod.BILINEAR),
                axis=-1),
            axis=-1)

        _action_prd = tf.stop_gradient(action * tf.expand_dims(
            (tf.argmax(_preact_prd_logits, axis=-1) + 1),
            axis=-1
        )
                                      )
        NS_action_prd = tf.stop_gradient(NS_action * tf.expand_dims(
            (tf.argmax(_preact_prd_logits, axis=-1) + 1),
            axis=-1
        )
                                         )

        self.dynamic_envs = self.__new_envs(_action_prd)

        return [state_value,
                policy,
                preact_interp_policy,
                interp_policy,
                action,
                NS_action,
                preact_interp_prd_logits,
                NS_action_interp_smooth]


    def __gen_action(self, policy):
        flat_policy = tf.reshape(policy, shape=(-1, 2))
        indice = tf.multinomial(logits=tf.log(flat_policy), num_samples=1)
        samples = tf.reshape(indice, shape=(-1,self.dynamic_envs_shape[1], self.dynamic_envs_shape[2], 1))

        return tf.stop_gradient(samples)


    def __new_envs(self, action_prd):

        return tf.stop_gradient(
            tf.to_float(
                tf.one_hot(
                    indices=tf.squeeze(
                        action_prd
                    ),
                    depth=self.total_classes
                )
            )
        )
        '''
        return tf.stop_gradient(
            tf.to_float(
                tf.one_hot(
                    indices=tf.squeeze(
                        layers.Interp(x=action,
                                      shape=self.dynamic_envs_shape_spatial,
                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                    ),
                    depth=self.total_classes
                )
            )
        )
        '''

    def __preprocess_image(self):
        """Preprocess an image as input."""
        self.processed_inputs = self.inputs - DATA_MEAN
        self.processed_inputs = self.processed_inputs[:, :, :, ::-1]  # RGB => BGR

        return True


    def __compute_reward(self, action, preact_interp_prd_logits, episode_step_count, action_shape):
        interp_action = layers.Interp(x=action,
                                      shape=self.inputs_shape_spatial,
                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        interp_action_prd = tf.stop_gradient(interp_action * tf.expand_dims(
            (tf.argmax(preact_interp_prd_logits, axis=-1) + 1),
            axis=-1
        )
                                             )
        certain_region_mask = x=tf.to_float(tf.not_equal(action, 0))


        uncertain_region_mask = x=tf.to_float(tf.equal(action, 0))


        uncertain_region_sum = uncertain_region_mask

        pos_mask = x=tf.to_float(tf.equal(interp_action_prd, self.ground_truth + 1))
        neg_mask = 1.0 - pos_mask
        pos_mask = layers.Interp(x=pos_mask, shape=action_shape)
        neg_mask = layers.Interp(x=neg_mask, shape=action_shape)

        unignored_mask = layers.Interp(x=tf.to_float(tf.not_equal(self.ground_truth, 255)),
                                       shape=action_shape,
                                       method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        pos_certreg_sum = pos_mask
        neg_certreg_sum = unignored_mask * certain_region_mask * neg_mask

        reward_dict_idx = episode_step_count if episode_step_count < (len(self.reward_dict)-1) else len(self.reward_dict)-1
        #reward = 1.0 * pos_certreg_sum - 1.0 * neg_certreg_sum - 0.1 * uncertain_region_sum
        reward = pos_certreg_sum - neg_certreg_sum + self.reward_dict[reward_dict_idx] * uncertain_region_sum

        return tf.stop_gradient(tf.squeeze(reward, axis=-1))


    def inference_chain(self, max_step=20):
        episode_NS_action_buffer = []
        episode_NS_action_interp_smooth_buffer = []
        episode_action_buffer = []
        episode_statevalue_buffer = []
        episode_reward_buffer = []
        episode_policy_buffer = []
        episode_preactinterppolicy_buffer = []
        episode_truereward_buffer = []
        episode_interppolicy_buffer = []
        episode_preactinterpprdlogits_buffer = []

        self.dynamic_envs = tf.zeros(shape=self.dynamic_envs_shape, dtype=tf.float32)
        for step in range(0, max_step+1):
            infos = self.__inference_once()

            state_value, policy, preact_interp_policy, interp_policy,\
            action, NS_action, preact_interp_prd_logits, NS_action_interp_smooth = infos

            tf.get_variable_scope().reuse_variables()
            reward = self.__compute_reward(action, preact_interp_prd_logits, step, self.dynamic_envs_shape_spatial)
            true_reward = self.__compute_reward(NS_action, preact_interp_prd_logits, step, self.dynamic_envs_shape_spatial)

            episode_NS_action_buffer.append(NS_action)
            episode_NS_action_interp_smooth_buffer.append(NS_action_interp_smooth)
            episode_preactinterpprdlogits_buffer.append(preact_interp_prd_logits)
            episode_action_buffer.append(action)
            episode_statevalue_buffer.append(state_value)
            episode_reward_buffer.append(reward)
            episode_policy_buffer.append(policy)
            episode_preactinterppolicy_buffer.append(preact_interp_policy)
            episode_interppolicy_buffer.append(interp_policy)
            episode_truereward_buffer.append(true_reward)

        self.episode_buffer = {'NSaction_buffer': episode_NS_action_buffer,
                               'NSactioninterpsmth_buffer': episode_NS_action_interp_smooth_buffer,
                               'preactinterpprdlogits_buffer':episode_preactinterpprdlogits_buffer,
                               'action_buffer':episode_action_buffer,
                               'statevalue_buffer':episode_statevalue_buffer,
                               'reward_buffer':episode_reward_buffer,
                               'policy_buffer':episode_policy_buffer,
                               'preactinterppolicy_buffer':episode_preactinterppolicy_buffer,
                               'truereward_buffer':episode_truereward_buffer,
                               'interppolicy_buffer': episode_interppolicy_buffer,
                               'prior_buffer': self.prior_logits}

        return self.episode_buffer


    def optimize(self, gamma=0.95):
        rollout = self.episode_buffer

        def _compute_loss(rollout, ignore_mask=False):
            preact_interp_prd_logits = tf.convert_to_tensor(rollout['preactinterpprdlogits_buffer'][0:])
            actions = tf.convert_to_tensor(rollout['action_buffer'][0:-1])
            values = tf.convert_to_tensor(rollout['statevalue_buffer'][0:-1])
            rewards = tf.convert_to_tensor(rollout['reward_buffer'][0:-1])
            policy = tf.convert_to_tensor(rollout['policy_buffer'][0:-1])
            bootstrap_value = tf.expand_dims(tf.convert_to_tensor(rollout['statevalue_buffer'][-1]), axis=0)

            rewards_plus = tf.concat([rewards, bootstrap_value], axis=0)
            discounted_rewards = tf.stop_gradient(discount(rewards_plus, gamma)[:-1])
            value_plus = tf.concat([values, bootstrap_value], axis=0)
            advantages = rewards + gamma * value_plus[1:] - value_plus[:-1]
            advantages = tf.stop_gradient(discount(advantages, gamma))
            #advantages = tf.expand_dims(tf.expand_dims(advantages, axis=-1), axis=-1)

            actions_onehot = tf.one_hot(tf.squeeze(actions), depth=2)
            responsible_outputs = tf.reduce_sum(policy * actions_onehot, axis=-1)

            value_loss = tf.reduce_sum(tf.square(discounted_rewards - values))
            policy_loss = -tf.reduce_sum(tf.log(responsible_outputs) * advantages)
            entropy_loss = tf.reduce_sum(policy * tf.log(policy))

            ground_truth_tile = tf.tile(
                tf.expand_dims(
                    self.ground_truth, axis=0),
                    [preact_interp_prd_logits.shape.as_list()[0], 1, 1, 1, 1]
            )
            modified_ground_truth = tf.to_float(
                tf.reshape(
                    tf.one_hot(tf.squeeze(ground_truth_tile), depth=self.nb_classes),
                    shape=[-1, self.nb_classes]
                )
            )
            if ignore_mask:
                ground_truth_unignore_mask = tf.reshape(
                    tf.to_float(tf.not_equal(ground_truth_tile, 255)),
                    shape=[-1]
                )
            else:
                ground_truth_unignore_mask = 1.0
            cls_loss = tf.reduce_sum(
                    tf.losses.softmax_cross_entropy(
                    onehot_labels=modified_ground_truth,
                    logits=tf.reshape(preact_interp_prd_logits, shape=[-1, self.nb_classes]),
                    weights=ground_truth_unignore_mask,
                    reduction=tf.losses.Reduction.NONE
                )
            )

            loss = value_loss + policy_loss + cls_loss + 0.3 * entropy_loss

            return loss, policy_loss, value_loss, entropy_loss, cls_loss

        self.loss, self.policy_loss, self.value_loss, self.entropy_loss, self.cls_loss = \
            _compute_loss(rollout, ignore_mask=True)

        return self.loss, self.policy_loss, self.value_loss, self.entropy_loss, self.cls_loss

