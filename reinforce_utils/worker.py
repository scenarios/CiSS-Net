import tensorflow as tf

from model_util.pspnet.pspnet import PSPNet
from model.psp import PSPNetworker, PSPNetworker_ade20k


class Worker(object):
    def __init__(self,
                 inputs,
                 ground_truth,
                 nb_classes=19,
                 resnet_layers=101,
                 num_step=3,
                 weights_path=None,
                 networker = 'ade20k',
                 is_training=True):
        self.inputs = inputs
        self.ground_truth = ground_truth
        self.num_step = num_step

        # Create the local copy of the network and the tensorflow op to copy global paramters to local network
        if networker == 'ade20k':
            self.networker = PSPNetworker_ade20k(
                nb_classes=nb_classes,
                inputs=self.inputs,
                ground_truth=self.ground_truth,
                weights_path=weights_path,
                is_training=is_training
            )
        else:
            self.networker = PSPNetworker(
                nb_classes=nb_classes,
                inputs=self.inputs,
                ground_truth=self.ground_truth,
                weights_path=weights_path,
                is_training=is_training
            )


    def __work(self):
        self.episode_buffer = self.networker.inference_chain(max_step=self.num_step)


    def __compute_loss(self):
        self.loss, self.policy_loss, self.value_loss, self.entropy_loss, self.cls_loss = self.networker.optimize()


    def work_with_loss(self):
        self.__work()
        self.__compute_loss()

        return self.loss, self.policy_loss, self.value_loss, self.entropy_loss, self.cls_loss


    @property
    def buffer(self):
        return {v : tf.convert_to_tensor(self.episode_buffer[v]) for v in self.episode_buffer.keys()}

    @property
    def prior(self):
        #return self.networker.prior_logits
        return None