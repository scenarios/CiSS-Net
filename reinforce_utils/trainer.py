import os

from reinforce_utils.helper.helpers import *
from reinforce_utils.worker import Worker
from evaluation import IoU_Evaluator

flags = tf.app.flags
flags.DEFINE_integer('max_steps', 20000000, 'Number of steps to run trainer.')
flags.DEFINE_string('trained_model_rtdir', 'D:/workspace/yizhou/train/rl-segmentation/trained_model', 'Trained model_util root directory')
flags.DEFINE_string('tensorboard_log_rtdir', 'D:/workspace/yizhou/train/rl-segmentation/log/tflog', 'tensorboard log root directory')
FLAGS = flags.FLAGS

class Trainer(object):
    def __init__(self,
                 session,
                 data_feeder,
                 optimizer,
                 num_steps=8,
                 max_steps=2000000,
                 num_gpus=8,
                 trained_model_rtdir='D:/workspace/yizhou/train/rl-segmentation/trained_model',
                 tensorboard_log_rtdir='D:/workspace/yizhou/train/rl-segmentation/log/tflog',
                 ft_model_path=None,
                 ft_var_list=None,
                 cnpt_model_path=None,
                 cnpt_var_list=None,
                 restore_model_path=None,
                 alias=None):
        self.sess = session
        self.restore_model_path = restore_model_path
        self.cnpt_model_path = cnpt_model_path
        self.cnpt_var_list = cnpt_var_list
        self.ft_model_path = ft_model_path
        self.ft_var_list = ft_var_list
        self.alias = alias

        self.log_step = 50
        self.do_validation_step = 500

        self.save_model_path = os.path.join(FLAGS.trained_model_rtdir, self.alias)

        if not os.path.exists(self.save_model_path):
            os.makedirs(self.save_model_path)

        self.optimizer = optimizer
        self.train_evaluator = IoU_Evaluator(num_class=150, is_training=True, dataset='ade20k')
        self.val_evaluator = IoU_Evaluator(num_class=150, is_training=True, dataset='ade20k')
        self.train_evaluator_cntpt = IoU_Evaluator(num_class=150, is_training=True, dataset='ade20k')
        self.val_evaluator_cntpt = IoU_Evaluator(num_class=150, is_training=True, dataset='ade20k')
        self.global_episodes = tf.get_variable('global_episodes',
                                               [],
                                               initializer=tf.constant_initializer(0),
                                               trainable=False)
        #self.increment = self.global_episodes.assign_add(1)

        self.num_step = num_steps


        #self.summary_writer = tf.summary.FileWriter("train_" + str(self.number))
        self.data_feeder_train, self.data_feeder_validation = data_feeder

        self.vis_image_pl = tf.placeholder(dtype=tf.float32, shape=[1, 473, 473, 3])
        self.vis_groundtruth_pl = tf.placeholder(dtype=tf.uint8, shape=[1, 473, 473, 3])
        self.vis_prior_pl = tf.placeholder(dtype=tf.uint8, shape=[13, 473, 473, 3])
        self.vis_prediction_pl = tf.placeholder(dtype=tf.uint8, shape=[3 * (self.num_step + 1), 473, 473, 3])
        self.base_summaries = [
            tf.summary.image('image_train', self.vis_image_pl),
            tf.summary.image('groundtruth_train', self.vis_groundtruth_pl),
            tf.summary.image('predictions_train', self.vis_prediction_pl, max_outputs=3 * (self.num_step + 1))
        ]

        self.workers = []
        self.global_gradient = []
        self.global_gradient_bottom = []
        self.global_gradient_top_plc = []
        self.global_gradient_top_cls = []
        self.global_gradient_top_bprd = []
        self.global_loss = []
        self.global_policy_loss = []
        self.global_value_loss = []
        self.global_entropy_loss = []
        self.global_cls_loss = []
        self.global_loss_validation = []
        self.global_policy_loss_validation = []
        self.global_value_loss_validation = []
        self.global_entropy_loss_validation = []
        self.global_cls_loss_validation = []

        self.episode_buffer = []
        self.episode_buffer_validation = []

    def setup_workers(self, worker_param=None):
        with tf.variable_scope(tf.get_variable_scope()) as _:
            for gpu_index in range(0, 3):
                with tf.device('/gpu:%d' % gpu_index):
                    with tf.name_scope('%s_%d' % ('gpu_', gpu_index)) as scope:
                        local_inputs, local_ground_truth = self.data_feeder_validation.dequeue()
                        local_worker = Worker(inputs=local_inputs,
                                              ground_truth=local_ground_truth,
                                              num_step=self.num_step,
                                              nb_classes=150)
                        self.workers.append(local_worker)
                        tf.get_variable_scope().reuse_variables()
            for gpu_index in range(3, 6):
                with tf.device('/gpu:%d' % gpu_index):
                    with tf.name_scope('%s_%d' % ('gpu_', gpu_index)) as scope:
                        local_inputs, local_ground_truth = self.data_feeder_train.dequeue()
                        local_worker = Worker(inputs=local_inputs,
                                              ground_truth=local_ground_truth,
                                              num_step=self.num_step,
                                              nb_classes=150)
                        self.workers.append(local_worker)
                        tf.get_variable_scope().reuse_variables()
            self.img_example = local_inputs[0]
            self.gt_example = local_ground_truth[0]
            # self.prior_logits = Interp(local_worker.prior[0], [713, 713])

            for gpu_index in range(6, 7):
                with tf.device('/gpu:%d' % gpu_index):
                    with tf.name_scope('%s_%d' % ('gpu_', gpu_index)) as scope:
                        local_inputs, local_ground_truth = self.data_feeder_validation.dequeue()
                        local_worker = Worker(inputs=local_inputs,
                                              ground_truth=local_ground_truth,
                                              num_step=self.num_step,
                                              nb_classes=150,
                                              is_training=False,)
                        self.workers.append(local_worker)
                        tf.get_variable_scope().reuse_variables()
            self.img_example_validation = local_inputs[0]
            self.gt_example_validation = local_ground_truth[0]


    def train(self):
        if not self.workers:
            self.setup_workers()
        with tf.variable_scope(tf.get_variable_scope()) as _:
            for gpu_index in range(0, 15):
                with tf.device('/gpu:%d' % gpu_index):
                    with tf.name_scope('%s_%d' % ('gpu_', gpu_index)) as scope:
                        local_loss, local_policy_loss, local_value_loss, local_entropy_loss, local_cls_loss = \
                            self.__compute_worker_loss(gpu_index)
                        buffer = self.__get_worker_buffer(gpu_index)

                        tf.get_variable_scope().reuse_variables()

                        local_trainable_var_top_plc = [v for v in tf.trainable_variables()
                                                       if 'top' in v.name and 'plc' in v.name]
                        local_trainable_var_top_cls = [v for v in tf.trainable_variables()
                                                       if 'top' in v.name and 'cls' in v.name]
                        local_trainable_var_top_bprd = [v for v in tf.trainable_variables()
                                                       if 'base_prediction' in v.name]

                        local_gradient_top_plc = \
                            self.optimizer['policy'].compute_gradients(loss=local_loss,
                                                                       var_list=local_trainable_var_top_plc)
                        local_gradient_top_cls = \
                            self.optimizer['classification'].compute_gradients(loss=local_loss,
                                                                               var_list=local_trainable_var_top_cls)
                        local_gradient_top_bprd = \
                            self.optimizer['base_prediction'].compute_gradients(loss=local_loss,
                                                                               var_list=local_trainable_var_top_bprd)
                        '''
                        local_gradient_bottom = \
                            self.optimizer[1].compute_gradients(loss=local_loss,
                                                                var_list=local_trainable_var_bottom)
                        '''
                        #local_gradient = local_gradient_top + local_gradient_bottom

                        self.episode_buffer.append(buffer)
                        #self.global_gradient.append(local_gradient)
                        self.global_gradient_top_plc.append(local_gradient_top_plc)
                        self.global_gradient_top_cls.append(local_gradient_top_cls)
                        self.global_gradient_top_bprd.append(local_gradient_top_bprd)
                        #self.global_gradient_bottom.append(local_gradient_bottom)
                        self.global_loss.append(local_loss)
                        self.global_policy_loss.append(local_policy_loss)
                        self.global_value_loss.append(local_value_loss)
                        self.global_entropy_loss.append(local_entropy_loss)
                        self.global_cls_loss.append(local_cls_loss)

            for gpu_index in range(15, 16):
                with tf.device('/gpu:%d' % gpu_index):
                    with tf.name_scope('%s_%d' % ('gpu_', gpu_index)) as scope:
                        local_loss, local_policy_loss, local_value_loss, local_entropy_loss, local_cls_loss = \
                            self.__compute_worker_loss(gpu_index)
                        buffer_validation = self.__get_worker_buffer(gpu_index)

                        tf.get_variable_scope().reuse_variables()

                        self.episode_buffer.append(buffer_validation)

                        self.global_loss_validation.append(local_loss)
                        self.global_policy_loss_validation.append(local_policy_loss)
                        self.global_value_loss_validation.append(local_value_loss)
                        self.global_entropy_loss_validation.append(local_entropy_loss)
                        self.global_cls_loss_validation.append(local_cls_loss)

        self.NSactioninterp_example = tf.reshape(tf.transpose(buffer['NSactioninterpsmth_buffer'], [1, 0, 2, 3, 4])[0],
                                                 [-1] +
                                                 buffer['NSactioninterpsmth_buffer'].shape.as_list()[2:])
        self.prd_example = tf.reshape(tf.transpose(buffer['preactinterpprdlogits_buffer'], [1, 0, 2, 3, 4])[0], [-1] +
                                      buffer['preactinterpprdlogits_buffer'].shape.as_list()[2:])
        self.pri_example = buffer['prior_buffer'][0]

        self.NSactioninterp_example_validation = \
            tf.reshape(tf.transpose(buffer_validation['NSactioninterpsmth_buffer'], [1, 0, 2, 3, 4])[0],
                       [-1] + buffer_validation['NSactioninterpsmth_buffer'].shape.as_list()[2:])
        self.prd_example_validation = \
            tf.reshape(tf.transpose(buffer_validation['preactinterpprdlogits_buffer'], [1, 0, 2, 3, 4])[0],
                       [-1] + buffer_validation['preactinterpprdlogits_buffer'].shape.as_list()[2:])
        self.pri_example_validation = buffer_validation['prior_buffer'][0]

        buffer_summaries = [
            tf.summary.histogram('action', buffer['action_buffer']),
            tf.summary.histogram('policy', buffer['policy_buffer'])
        ]

        summaries = buffer_summaries + self.base_summaries
        summary_op = tf.summary.merge(summaries)

        grads = self.__average_gradients()
        losses = self.__average_loss(scope='training')
        losses_validation = self.__average_loss(scope='validation')

        assert len(self.optimizer) == len(grads), 'Number of optimizers does not match'

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            apply_grads_op_cls = self.optimizer['classification'].apply_gradients(
                grads_and_vars=grads['classification'], global_step=self.global_episodes
            )
            apply_grads_op_plc = self.optimizer['policy'].apply_gradients(
                grads_and_vars=grads['policy'], global_step=self.global_episodes
            )
            apply_grads_op_bprd = self.optimizer['base_prediction'].apply_gradients(
                grads_and_vars=grads['base_prediction'], global_step=self.global_episodes
            )
            apply_grads_op_all = tf.group(apply_grads_op_cls, apply_grads_op_plc, apply_grads_op_bprd)
        '''
        with tf.control_dependencies(update_ops):
            apply_grads_op = tf.group(
                *[self.optimizer[idx].apply_gradients(grads_and_vars=grads[idx], global_step=self.global_episodes)
                 for idx in range(0, len(grads))]
            )
        '''
        _ft_var_list = [v for v in tf.global_variables() if v.name in self.ft_var_list]
        self.global_train_saver = tf.train.Saver(max_to_keep=20)
        self.ft_saver = tf.train.Saver(var_list=_ft_var_list)
        #self.cnpt_saver = tf.train.Saver(var_list=[v for v in tf.global_variables(scope='cnpt') if v.name in self.cnpt_var_list])

        global_var_init_op = tf.global_variables_initializer()
        self.sess.run(global_var_init_op)

        if self.restore_model_path:
            print("Restore all variables and continue training.")
            self.global_train_saver.restore(sess=self.sess, save_path=self.restore_model_path)
            #self.cnpt_saver.restore(sess=self.sess, save_path=self.cnpt_model_path)
        elif self.ft_model_path and self.ft_saver:
            print("Training with pretrained model_util.")
            self.ft_saver.restore(sess=self.sess, save_path=self.ft_model_path)
            #self.cnpt_saver.restore(sess=self.sess, save_path=self.cnpt_model_path)
        else:
            print("Training from the scratch.")
            #self.cnpt_saver.restore(sess=self.sess, save_path=self.cnpt_model_path)

        self.summary_writer = \
            tf.summary.FileWriter(os.path.join(FLAGS.tensorboard_log_rtdir, self.alias), self.sess.graph)
        tf.train.start_queue_runners(sess=self.sess)
        global_step = 0 if not self.restore_model_path else int(self.sess.run(self.global_episodes)/2)
        try:

            def _loss_acc_update(loss_value, scope, train_loss_acc, val_loss_acc):
                if scope == 'train':
                    for i in range(len(train_loss_acc)):
                        train_loss_acc[i] += loss_value[i]
                elif scope == 'validation':
                    for i in range(len(val_loss_acc)):
                        val_loss_acc[i] += loss_value[i]
                else:
                    print('Warning: Not valid scope name %s' % scope)

            #for step in range(init_step, 1000000):
            def _local_run(apply_grads_op, step, local_step, train_loss_acc, val_loss_acc):
                loss_value, buffer_value, _, img_v, gt_v, prd_v, NSactitp_v, pri_v, validations = \
                    self.sess.run([losses,
                                   [buffer['statevalue_buffer'], buffer['reward_buffer'], buffer['truereward_buffer']],
                                   apply_grads_op,
                                   self.img_example,
                                   self.gt_example,
                                   self.prd_example,
                                   self.NSactioninterp_example,
                                   self.pri_example,
                                   [losses_validation,
                                    self.gt_example_validation,
                                    self.prd_example_validation,
                                    self.pri_example_validation,
                                    self.NSactioninterp_example_validation,
                                    [
                                        buffer_validation['statevalue_buffer'],
                                        buffer_validation['reward_buffer'],
                                        buffer_validation['truereward_buffer']
                                    ]
                                    ]
                                   ]
                                  )
                print(loss_value[0], loss_value[1], loss_value[2], loss_value[3], loss_value[4])

                loss_validation_value, gt_validation_v, prd_validation_v, pri_validation_v, NSactitp_validation_v = \
                    validations[0:-1]
                buffer_validation_value = validations[-1]

                _loss_acc_update(loss_value, 'train', train_loss_acc, val_loss_acc)
                _loss_acc_update(loss_validation_value, 'validation',train_loss_acc, val_loss_acc)
                # gt_v -= 1
                gt_v = np.squeeze(gt_v.astype(dtype=np.uint8), axis=-1)
                gt_validation_v = np.squeeze(gt_validation_v.astype(dtype=np.uint8), axis=-1)
                self.train_evaluator.evaluate(prediction=prd_v[-1], ground_truth=gt_v)
                self.train_evaluator_cntpt.evaluate(prediction=pri_v, ground_truth=gt_v)
                self.val_evaluator.evaluate(prediction=prd_validation_v[-1], ground_truth=gt_validation_v)
                self.val_evaluator_cntpt.evaluate(prediction=pri_validation_v, ground_truth=gt_validation_v)

                if local_step % self.log_step == 0:

                    state_value_mean = np.mean(buffer_value[0])
                    reward_value_mean = np.mean(buffer_value[1])
                    truereward_value_mean = np.mean(buffer_value[2])

                    summary = tf.Summary()

                    def __add_train_summary():
                        summary.value.add(tag='Reward_Mean', simple_value=float(reward_value_mean))
                        summary.value.add(tag='StateValue_Mean', simple_value=float(state_value_mean))
                        summary.value.add(tag='TotalLoss_Mean', simple_value=float(train_loss_acc[0]/self.log_step))
                        summary.value.add(tag='PolicyLoss_Mean', simple_value=float(train_loss_acc[1]/self.log_step))
                        summary.value.add(tag='ValueLoss_Mean', simple_value=float(train_loss_acc[2]/self.log_step))
                        summary.value.add(tag='EntropyLoss_Mean', simple_value=float(train_loss_acc[3]/self.log_step))
                        summary.value.add(tag='ClassificationLoss_Mean', simple_value=float(train_loss_acc[4]/self.log_step))
                        summary.value.add(tag='TrueReward_Mean', simple_value=float(truereward_value_mean))

                        for key in self.train_evaluator.IoU.keys():
                            this_IoU = float(self.train_evaluator.IoU[key])
                            summary.value.add(tag='IoU_train_'+key, simple_value=this_IoU)

                        for key in self.train_evaluator_cntpt.IoU.keys():
                            this_IoU = float(self.train_evaluator_cntpt.IoU[key])
                            summary.value.add(tag='IoU_train_cntpt_' + key, simple_value=this_IoU)

                    __add_train_summary()
                    self.summary_writer.add_summary(summary, step//self.log_step)
                    self.summary_writer.flush()

                    colored_prd_images, colored_gt_image = \
                        self.train_evaluator.visualization(prd_v=prd_v, NSactitp_v=NSactitp_v, gt_v=gt_v)

                    img_v  = np.expand_dims(img_v, axis=0)
                    summary_str = self.sess.run(summary_op,
                                                feed_dict={
                                                    self.vis_image_pl:img_v,
                                                    self.vis_groundtruth_pl:colored_gt_image,
                                                    #self.vis_prior_pl:colored_prd_images,
                                                    self.vis_prediction_pl:colored_prd_images
                                                })
                    self.summary_writer.add_summary(summary_str, step // self.log_step)
                    self.summary_writer.flush()

                    for i in range(len(train_loss_acc)):
                        train_loss_acc[i] = 0

                    self.train_evaluator.empty()
                    self.train_evaluator_cntpt.empty()

                if local_step % self.do_validation_step == 0:

                    state_value_validation_mean = np.mean(buffer_validation_value[0])
                    reward_value_validation_mean = np.mean(buffer_validation_value[1])
                    truereward_value_validation_mean = np.mean(buffer_validation_value[2])

                    summary = tf.Summary()

                    def __add_validation_summary():
                        summary.value.add(tag='Reward_Validation_Mean', simple_value=float(reward_value_validation_mean))
                        summary.value.add(tag='StateValue_Validation_Mean', simple_value=float(state_value_validation_mean))
                        summary.value.add(tag='TotalLoss_Validation_Mean', simple_value=float(val_loss_acc[0]/self.do_validation_step))
                        summary.value.add(tag='PolicyLoss_Validation_Mean', simple_value=float(val_loss_acc[1]/self.do_validation_step))
                        summary.value.add(tag='ValueLoss_Validation_Mean', simple_value=float(val_loss_acc[2]/self.do_validation_step))
                        summary.value.add(tag='EntropyLoss_Validation_Mean', simple_value=float(val_loss_acc[3]/self.do_validation_step))
                        summary.value.add(tag='ClassificationLoss_Validation_Mean', simple_value=float(val_loss_acc[4]/self.do_validation_step))
                        summary.value.add(tag='TrueReward_Validation_Mean', simple_value=float(truereward_value_validation_mean))

                        for key in self.val_evaluator.IoU.keys():
                            this_IoU = float(self.val_evaluator.IoU[key])
                            summary.value.add(tag='IoU_validation_' + key, simple_value=this_IoU)

                        for key in self.val_evaluator_cntpt.IoU.keys():
                            this_IoU = float(self.val_evaluator_cntpt.IoU[key])
                            summary.value.add(tag='IoU_validation_cntpt_' + key, simple_value=this_IoU)

                    __add_validation_summary()
                    self.summary_writer.add_summary(summary, step // self.log_step)
                    self.summary_writer.flush()

                    for i in range(len(val_loss_acc)):
                        val_loss_acc[i] = 0

                    self.val_evaluator.empty()
                    self.val_evaluator_cntpt.empty()
                    '''
                    colored_prd_images_validation, colored_gt_image_validation = \
                        self.train_evaluator.visualization(prd_v=prd_validation_v,
                                                           NSactitp_v=NSactitp_validation_v,
                                                           gt_v=gt_validation_v)

                    img_v = np.expand_dims(img_v, axis=0)
                    summary_str = self.sess.run(summary_op,
                                                feed_dict={
                                                    self.vis_image_pl: img_v,
                                                    self.vis_groundtruth_pl: colored_gt_image,
                                                    # self.vis_prior_pl:colored_prd_images,
                                                    self.vis_prediction_pl: colored_prd_images
                                                })
                    self.summary_writer.add_summary(summary_str, step // log_step)
                    self.summary_writer.flush()
                    '''


                if local_step % 1000 == 0:
                    glb_eps = int(self.sess.run(self.global_episodes))
                    sv_pth = self.global_train_saver.save(self.sess,
                                                    os.path.join(self.save_model_path, 'rl-segment' + str(glb_eps) + '.ckpt'))
                    print("Checkpoint model_util saved to " + sv_pth)

            train_loss_acc = [0, 0, 0, 0, 0]
            val_loss_acc = [0, 0, 0, 0, 0]
            for local_step in range(0, 3000000):
                _local_run(
                    apply_grads_op=apply_grads_op_all,
                    step=global_step, local_step=local_step+1,
                    train_loss_acc=train_loss_acc, val_loss_acc=val_loss_acc)
                global_step += 1

        except KeyboardInterrupt:
            print("Manual interrupt occurred.")
            glb_eps = int(self.sess.run(self.global_episodes))
            print('Done training for {} steps'.format(global_step))
            print('Done training for {} global steps'.format(glb_eps))
            print("Saving model_util...")
            sv_pth = self.global_train_saver.save(self.sess, os.path.join(self.save_model_path, 'rl-segment'+str(glb_eps)+'.ckpt'))
            print("Model saved to " + sv_pth)

    def tester(self):
        import cv2
        from scipy import misc
        import model_util.pspnet.utils as utils
        from model_util.pspnet.layers_builder import Interp
        from model_util.pspnet.layers_utils import Activation

        if not self.workers:
            self.setup_workers()
        tester = self.workers[0].networker.prior_logits
        tester = Interp(tester, [473, 473])
        tester = Activation(tester, 'softmax')
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, '../model_util/pspnet/weights/test.ckpt')
            cap = cv2.VideoCapture('../model_util/pspnet/example_images/ade20k.jpg')
            ret, img = cap.read()

            # img = cv2.resize(img,(int(16.0*713/9.0),713))
            img_orig = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img_orig, (473, 473), interpolation=cv2.INTER_AREA)
            img = np.asarray(img, dtype=np.float32)
            img = img[np.newaxis, :, :, :]
            img = np.concatenate((img, img), axis=0)
            class_scores = sess.run(tester,
                                    feed_dict={
                                        self.workers[0].inputs: img
                                    })
            # class_scores = cv2.resize(class_scores[0, :, :, :], (h_ori, w_ori))
            # End time
            class_scores = class_scores[0, :, :, :]

            full_probs = np.zeros((img_orig.shape[0], img_orig.shape[1], 150))
            h_ori, w_ori = img_orig.shape[:2]
            # scale probs up to full size
            probs = cv2.resize(class_scores, (w_ori, h_ori))
            # visualize_prediction(probs)
            # integrate probs over all scales
            class_scores = full_probs + probs

            class_image = np.argmax(class_scores, axis=2)
            pm = np.max(class_scores, axis=2)
            colored_class_image = utils.color_class_image(class_image,'pspnet50_ade20k')

            alpha_blended = 0.5 * colored_class_image + 0.5 * img_orig


            # cv2.putText(alpha_blended,'PSPNet Prediction time: %.0fms (%.1f fps) AVG: %.0fms (%.1f fps)'%(diff.microseconds/1000.0,1000000.0/diff.microseconds,time_sum/(counter+1),1000.0/(time_sum/(counter+1))),(100,100), cv2.FONT_HERSHEY_SIMPLEX, 3,(0,0,0),16,cv2.LINE_AA)
            # cv2.putText(alpha_blended,'PSPNet Prediction time: %.0fms (%.1f fps) AVG: %.0fms (%.1f fps)'%(diff.microseconds/1000.0,1000000.0/diff.microseconds,time_sum/(counter+1),1000.0/(time_sum/(counter+1))),(100,100), cv2.FONT_HERSHEY_SIMPLEX, 3,(255,255,255),10,cv2.LINE_AA)

            misc.imsave('../model_util/pspnet/example_results/testerc.jpg', colored_class_image)
            misc.imsave('../model_util/pspnet/example_results/testera.jpg', alpha_blended)


    def __compute_worker_loss(self, worker_id):
        return self.workers[worker_id].work_with_loss()


    def __get_worker_buffer(self, worker_id):
        return self.workers[worker_id].buffer


    def __average_gradients(self):
        """Calculate the average gradient for each shared variable across all towers.

        Note that this function provides a synchronization point across all towers.

        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been averaged
           across all towers.
        """
        averaged_grads = {}
        if self.global_gradient:
            _average_grads = []
            for grad_and_vars in zip(*self.global_gradient):
                # Note that each grad_and_vars looks like the following:
                #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
                grads = []
                for g, _ in grad_and_vars:
                    # Add 0 dimension to the gradients to represent the tower.
                    expanded_g = tf.expand_dims(g, 0)

                    # Append on a 'tower' dimension which we will average over below.
                    grads.append(expanded_g)

                # Average over the 'tower' dimension.
                grad = tf.concat(axis=0, values=grads)
                grad = tf.reduce_mean(grad, 0)

                # Keep in mind that the Variables are redundant because they are shared
                # across towers. So .. we will just return the first tower's pointer to
                # the Variable.
                v = grad_and_vars[0][1]
                grad_and_var = (grad, v)
                _average_grads.append(grad_and_var)
            averaged_grads['global_gradient'] = _average_grads
        else:
            _average_grads_top_plc = []
            _average_grads_top_cls = []
            _average_grads_top_bprd = []
            _average_grads_bottom = []
            for grad_and_vars in zip(*self.global_gradient_top_plc):
                # Note that each grad_and_vars looks like the following:
                #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
                grads = []
                for g, _ in grad_and_vars:
                    if g is not None:
                        # Add 0 dimension to the gradients to represent the tower.
                        expanded_g = tf.expand_dims(g, 0)

                        # Append on a 'tower' dimension which we will average over below.
                        grads.append(expanded_g)
                if grads:
                    # Average over the 'tower' dimension.
                    grad = tf.concat(axis=0, values=grads)
                    grad = tf.reduce_mean(grad, 0)

                    # Keep in mind that the Variables are redundant because they are shared
                    # across towers. So .. we will just return the first tower's pointer to
                    # the Variable.
                    v = grad_and_vars[0][1]
                    grad_and_var = (grad, v)
                    _average_grads_top_plc.append(grad_and_var)

            for grad_and_vars in zip(*self.global_gradient_top_cls):
                # Note that each grad_and_vars looks like the following:
                #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
                grads = []
                for g, _ in grad_and_vars:
                    if g is not None:
                        # Add 0 dimension to the gradients to represent the tower.
                        expanded_g = tf.expand_dims(g, 0)

                        # Append on a 'tower' dimension which we will average over below.
                        grads.append(expanded_g)
                if grads:
                    # Average over the 'tower' dimension.
                    grad = tf.concat(axis=0, values=grads)
                    grad = tf.reduce_mean(grad, 0)

                    # Keep in mind that the Variables are redundant because they are shared
                    # across towers. So .. we will just return the first tower's pointer to
                    # the Variable.
                    v = grad_and_vars[0][1]
                    grad_and_var = (grad, v)
                    _average_grads_top_cls.append(grad_and_var)

            for grad_and_vars in zip(*self.global_gradient_top_bprd):
                # Note that each grad_and_vars looks like the following:
                #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
                grads = []
                for g, _ in grad_and_vars:
                    if g is not None:
                        # Add 0 dimension to the gradients to represent the tower.
                        expanded_g = tf.expand_dims(g, 0)

                        # Append on a 'tower' dimension which we will average over below.
                        grads.append(expanded_g)
                if grads:
                    # Average over the 'tower' dimension.
                    grad = tf.concat(axis=0, values=grads)
                    grad = tf.reduce_mean(grad, 0)

                    # Keep in mind that the Variables are redundant because they are shared
                    # across towers. So .. we will just return the first tower's pointer to
                    # the Variable.
                    v = grad_and_vars[0][1]
                    grad_and_var = (grad, v)
                    _average_grads_top_bprd.append(grad_and_var)

            for grad_and_vars in zip(*self.global_gradient_bottom):
                # Note that each grad_and_vars looks like the following:
                #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
                grads = []
                for g, _ in grad_and_vars:
                    if g is not None:
                        # Add 0 dimension to the gradients to represent the tower.
                        expanded_g = tf.expand_dims(g, 0)

                        # Append on a 'tower' dimension which we will average over below.
                        grads.append(expanded_g)

                if grads:
                    # Average over the 'tower' dimension.
                    grad = tf.concat(axis=0, values=grads)
                    grad = tf.reduce_mean(grad, 0)

                    # Keep in mind that the Variables are redundant because they are shared
                    # across towers. So .. we will just return the first tower's pointer to
                    # the Variable.
                    v = grad_and_vars[0][1]
                    grad_and_var = (grad, v)
                    _average_grads_bottom.append(grad_and_var)

            if _average_grads_top_plc:
                averaged_grads['policy'] = _average_grads_top_plc
            if _average_grads_top_cls:
                averaged_grads['classification'] = _average_grads_top_cls
            if _average_grads_top_bprd:
                averaged_grads['base_prediction'] = _average_grads_top_bprd
            if _average_grads_bottom:
                averaged_grads['bottom'] = _average_grads_bottom

        return averaged_grads


    def __average_loss(self, scope='training'):
        assert scope in ['training', 'validation'], 'Unknown scope' + scope
        if scope is 'training':
            assert  self.global_loss, 'Error: None global loss.'
            assert  self.global_policy_loss, 'Error: None global policy loss.'
            assert  self.global_value_loss, 'Error: None global value loss.'
            assert  self.global_entropy_loss, 'Error: None global entropy loss.'
            assert  self.global_cls_loss, 'Error: None global classification loss.'

            global_loss_ave = sum(self.global_loss) / len(self.global_loss)
            global_policy_loss_ave = sum(self.global_policy_loss) / len(self.global_policy_loss)
            global_value_loss_ave = sum(self.global_value_loss) / len(self.global_value_loss)
            global_entropy_loss_ave = sum(self.global_entropy_loss) / len(self.global_entropy_loss)
            global_cls_loss_ave = sum(self.global_cls_loss) / len(self.global_cls_loss)

            return [global_loss_ave,
                    global_policy_loss_ave,
                    global_value_loss_ave,
                    global_entropy_loss_ave,
                    global_cls_loss_ave]

        elif scope is 'validation':
            assert self.global_loss_validation, 'Error: None global validation loss.'
            assert self.global_policy_loss_validation, 'Error: None global policy validation loss.'
            assert self.global_value_loss_validation, 'Error: None global value validation loss.'
            assert self.global_entropy_loss_validation, 'Error: None global entropy validation loss.'
            assert self.global_cls_loss_validation, 'Error: None global classification validation loss.'

            global_loss_ave = sum(self.global_loss_validation) / len(self.global_loss_validation)
            global_policy_loss_ave = sum(self.global_policy_loss_validation) / len(self.global_policy_loss_validation)
            global_value_loss_ave = sum(self.global_value_loss_validation) / len(self.global_value_loss_validation)
            global_entropy_loss_ave = sum(self.global_entropy_loss_validation) / len(self.global_entropy_loss_validation)
            global_cls_loss_ave = sum(self.global_cls_loss_validation) / len(self.global_cls_loss_validation)


            return [global_loss_ave,
                    global_policy_loss_ave,
                    global_value_loss_ave,
                    global_entropy_loss_ave,
                    global_cls_loss_ave]


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 2, 4"
    trainer = Trainer()
    trainer.train()