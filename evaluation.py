import numpy as np

import model_util.pspnet.utils as utils


class Evaluator(object):
    def __init__(self, dataset, is_training):
        super(Evaluator, self).__init__()
        self.is_training = is_training
        self.dataset = dataset

    def evaluate(self, prediction, ground_truth):
        pass

    def visualization(self, pri_v, prd_v, gt_v):
        pass

    def empty(self):
        pass


class IoU_Evaluator(Evaluator):
    def __init__(self, num_class, dataset, is_training, max_buffer_size=100000):
        super(IoU_Evaluator, self).__init__(dataset, is_training)
        self.num_class = num_class
        self.threshold_dict = {0: 0.99,
                               1: 0.95,
                               2: 0.90,
                               3: 0.85,
                               4: 0.80,
                               5: 0.75,
                               6: 0.70,
                               7: 0.60,
                               8: 0.50,
                               9: 0.40,
                               10: 0.30
                               }
        self._IoU_dict = {}
        self._iIoU_dict = {}
        self._IoU_acc = {}
        self._iIoU_acc = {}
        self._IoU_buffer = {}
        self._iIoU_buffer = {}
        self._max_buffer_size = max_buffer_size
        self._count = {}

    def __update(self):
        if self.is_training:
            for key in self._IoU_dict.keys():
                if self._IoU_dict[key] == -1:
                    pass
                else:
                    if key not in self._IoU_buffer:
                        self._IoU_buffer[key] = [self._IoU_dict[key]]
                    else:
                        if len(self._IoU_buffer[key]) < self._max_buffer_size:
                            self._IoU_buffer[key].append(self._IoU_dict[key])
                        else:
                            self._IoU_buffer[key].pop(0)
                            self._IoU_buffer[key].append(self._IoU_dict[key])
        else:
            for key in self._IoU_dict.keys():
                if self._IoU_dict[key] == -1:
                    pass
                else:
                    self._count[key] = \
                        self._count[key] + 1 if key in self._count else 0
                    self._IoU_acc[key] = \
                        self._IoU_acc[key] + self._IoU_dict[key] if key in self._IoU_acc else self._IoU_dict[key]
        self._IoU_dict = {}

    def __evaluate_once(self, class_prd_nuc, ground_truth):
        unignored_region = ground_truth != 255
        for i in range(self.num_class):
            this_predicted_positive_region = class_prd_nuc == i
            this_groudtruth_positive_region = ground_truth == i
            TP_region = np.logical_and(this_predicted_positive_region, this_groudtruth_positive_region)
            TPFPFN_region = np.logical_or(this_predicted_positive_region, this_groudtruth_positive_region)
            TP = np.sum(np.logical_and(TP_region, unignored_region))
            TPFPFN = np.sum(np.logical_and(TPFPFN_region, unignored_region))

            self._IoU_dict[utils.ade20k_id2label[i].name] = TP / TPFPFN if TPFPFN > 0 else -1

    def evaluate(self, prediction, ground_truth):
        assert prediction.shape[-1] - self.num_class == 0, "number of predicted class mismatch"
        def __preprocess():
            class_prd = np.argmax(prediction, axis=-1)
            # nuc -> none uncertainty
            class_prd_nuc = np.copy(class_prd)
            class_prd_nuc[class_prd_nuc == -1] = \
                np.argsort(-prediction, axis=-1)[:, :, 1][class_prd_nuc == -1]
            return class_prd_nuc

        class_prd_nuc = __preprocess()
        self.__evaluate_once(class_prd_nuc, ground_truth)
        self.__update()

    def empty(self):
        self._IoU_buffer = {}

    def visualization(self, prd_v, NSactitp_v, gt_v, pri_v=None):
        threshold_dict = {0: 0.99,
                          1: 0.95,
                          2: 0.90,
                          3: 0.85,
                          4: 0.80,
                          5: 0.75,
                          6: 0.70,
                          7: 0.60,
                          8: 0.50,
                          9: 0.40,
                          10: 0.30
                          }
        def __compute_colored_gt_image():
            colored_gt_image = utils.color_class_image(gt_v, self.dataset)
            colored_gt_image = np.expand_dims(colored_gt_image, axis=0)

            return colored_gt_image

        def __compute_colored_pri_image():
            class_image_prior = np.argmax(pri_v, axis=-1)
            class_image_prior[gt_v == 255] = 255
            class_image_prior_neq = np.copy(class_image_prior)
            class_image_prior_neq[class_image_prior_neq != gt_v] = -1
            probability_image_prior = np.max(pri_v, axis=-1)
            class_image_prior_threshold = [class_image_prior, class_image_prior_neq]
            for i in range(0, 11):
                class_image_prior_copy = np.copy(class_image_prior)
                class_image_prior_copy[probability_image_prior < threshold_dict[i]] = -1
                class_image_prior_copy[gt_v == 255] = 255
                class_image_prior_threshold.append(class_image_prior_copy)
            colored_pri_images = []
            for i in range(0, 13):
                this_class_image_prior = class_image_prior_threshold[i]
                colored_pri_images.append(
                    utils.color_class_image(this_class_image_prior, self.dataset))
            colored_pri_images = np.asarray(colored_pri_images, dtype=np.uint8)

            return colored_pri_images

        def __compute_colored_prd_image():
            class_image_prd_nuc = np.argmax(prd_v, axis=-1)
            actprd =np.squeeze(
                NSactitp_v * (np.expand_dims(class_image_prd_nuc, axis=-1) + 1) - 1,
                axis=-1
            )

            colored_prd_images = []
            for i in range(0, actprd.shape[0]):
                this_class_image_prd_nuc = class_image_prd_nuc[i]
                this_class_image_prd_nuc[gt_v == 255] = 255
                colored_prd_images.append(
                    utils.color_class_image(this_class_image_prd_nuc, self.dataset))

                this_class_image_prd_nuc_neq = np.copy(this_class_image_prd_nuc)
                this_class_image_prd_nuc_neq[this_class_image_prd_nuc_neq != gt_v] = -1
                this_class_image_prd_nuc_neq[gt_v == 255] = 255
                colored_prd_images.append(
                    utils.color_class_image(this_class_image_prd_nuc_neq, self.dataset))

                this_class_image_actprd = actprd[i]
                this_class_image_actprd[gt_v == 255] = 255
                colored_prd_images.append(
                    utils.color_class_image(this_class_image_actprd, self.dataset))
            colored_prd_images = np.asarray(colored_prd_images, dtype=np.uint8)

            return colored_prd_images

        colored_gt_image = __compute_colored_gt_image()
        colored_prd_images = __compute_colored_prd_image()
        if pri_v is not None:
            colored_pri_images = __compute_colored_pri_image()
            return colored_pri_images, colored_prd_images, colored_gt_image
        else:
            return colored_prd_images, colored_gt_image

    @property
    def IoU(self):
        IoU_mean = {'class_mean':[]}
        if self.is_training:
            for key in self._IoU_buffer.keys():
                IoU_mean[key] = sum(self._IoU_buffer[key]) / len(self._IoU_buffer[key])
                IoU_mean['class_mean'] += self._IoU_buffer[key]
        else:
            for key in self._IoU_buffer.keys():
                IoU_mean[key] = self._IoU_acc[key] / self._count[key]
                IoU_mean['class_mean'] += self._IoU_buffer[key]
        IoU_mean['class_mean'] = sum(IoU_mean['class_mean']) / len(IoU_mean['class_mean'])
        return IoU_mean