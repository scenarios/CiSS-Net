import os
import argparse
import tensorflow as tf

from reinforce_utils.trainer import Trainer
from datasets.data import batch_for_train, batch_for_validation
from tools.ckpt_parser import obtain_tensors_in_checkpoint_file


def training():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ,10, 11, 12, 13, 14, 15"
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = False
        sess = tf.Session(config=config)

        optimizer_top_cls = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
        optimizer_top_plc = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate * FLAGS.plc_learning_rate_ratio)
        optimizer_base_prediction = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate * FLAGS.bprd_learning_rate_ratio)
        #optimizer_bottom = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate * FLAGS.learning_rate_ration)
        optimizer = {'classification': optimizer_top_cls, 'policy': optimizer_top_plc, 'base_prediction': optimizer_base_prediction}
        images_train, labels_train = batch_for_train(
            filename='C:/Users/v-yizzh/Documents/code/rl-segmentation/datasets/ade20k_train_list.txt',
            batch_size=FLAGS.batch_size
        )
        images_validation, labels_validation = batch_for_validation(
            filename='C:/Users/v-yizzh/Documents/code/rl-segmentation/datasets/ade20k_validation_list.txt',
            batch_size=FLAGS.batch_size
        )

        batch_queue_train = tf.contrib.slim.prefetch_queue.prefetch_queue(
            [images_train, labels_train], capacity=4 * 7)
        batch_queue_validation = tf.contrib.slim.prefetch_queue.prefetch_queue(
            [images_validation, labels_validation], capacity=4 * 7)

        ft_model_path = "C:/Users/v-yizzh/Documents/code/rl-segmentation/model_util/pspnet/weights/pspnet50_ade20k.ckpt"
        ft_var_list = obtain_tensors_in_checkpoint_file(ft_model_path)
        cnpt_model_path = "C:/Users/v-yizzh/Documents/code/rl-segmentation/model_util/pspnet/weights/pspnet50_ade20k.ckpt"
        cnpt_var_list = obtain_tensors_in_checkpoint_file(cnpt_model_path)

        trainer = Trainer(session=sess,
                          optimizer=optimizer,
                          data_feeder=[batch_queue_train, batch_queue_validation],
                          ft_model_path=ft_model_path,
                          ft_var_list=ft_var_list,
                          cnpt_model_path=cnpt_model_path,
                          cnpt_var_list=cnpt_var_list,
                          num_steps=FLAGS.num_steps,
                          max_steps=FLAGS.max_steps,
                          num_gpus=FLAGS.num_gpus,
                          trained_model_rtdir=FLAGS.trained_model_root_dir,
                          tensorboard_log_rtdir=FLAGS.tensorboard_log_root_dir,
                          restore_model_path="D:/workspace/yizhou/train/rl-segmentation/trained_model/ade20k/Inc_softenv_T8_LOSS_1-1-1-005_REWARD_new_old_combined/LR_1e-11_1e-11_3e-11_sgd_FT_XX_OTHER_xx/rl-segment335459.ckpt",
                          alias="ade20k/Inc_softenv_T8_LOSS_1-1-1-005_REWARD_new_old_combined/LR_1e-11_1e-11_3e-11_sgd_FT_XX_OTHER_xx")

        trainer.train()

if __name__ == '__main__':
    flags = argparse.ArgumentParser()
    flags.add_argument('--batch_size', type=int, default=2)
    flags.add_argument('--learning_rate', type=float, default=1e-11)
    flags.add_argument('--plc_learning_rate_ratio', type=float, default=1.0)
    flags.add_argument('--bprd_learning_rate_ratio', type=float, default=3.0)
    flags.add_argument('--num_gpus', type=int, default=8)
    flags.add_argument('--max_steps', type=int, default=2000000)
    flags.add_argument('--num_steps', type=int, default=8)
    flags.add_argument('--trained_model_root_dir', default='D:/workspace/yizhou/train/rl-segmentation/trained_model')
    flags.add_argument('--tensorboard_log_root_dir', default='D:/workspace/yizhou/train/rl-segmentation/log/tflog')

    FLAGS = flags.parse_args()
    training()