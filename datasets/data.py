# coding=utf-8
import os
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from PIL import Image
import numpy as np


def generate_input_string_queue(filename):
    lines = list(open(filename, 'r'))
    img_gt_names = []
    for index in range(0, len(lines)):
        line = lines[index].strip('\n').split()
        img_name = line[0]
        gt_name = line[1]
        img_gt_name = '-'.join([img_name, gt_name])
        img_gt_names.append(img_gt_name)

    return tf.train.string_input_producer(img_gt_names, capacity=100000, shuffle=True)


def _compute_longer_edge(edge_a, edge_b, new_edge):
    tf.cast(edge_a, tf.float32)
    tf.cast(edge_b, tf.float32)
    tf.cast(new_edge, tf.float32)
    return tf.cast((edge_b * new_edge / edge_a), tf.int32)


def _resize_image_on_shorter_edge(image, new_shorter_edge_size, method):
    new_shorter_edge = tf.constant(new_shorter_edge_size, dtype=tf.int32)
    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]

    height_smaller_than_width = tf.less_equal(height, width)
    new_height_and_width = tf.cond(
        height_smaller_than_width,
        lambda: (new_shorter_edge, _compute_longer_edge(height, width, new_shorter_edge)),
        lambda: (_compute_longer_edge(width, height, new_shorter_edge), new_shorter_edge)
    )

    return tf.image.resize_images(image, new_height_and_width, method=method)

def single_image_preprocess(image, is_training, color_order=None):
    if image.dtype != tf.float32:
        # image = tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)
        image = tf.to_float(image)

    if is_training:
        image = tf.image.random_brightness(image, max_delta=16. / 255.)
        #image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
        #image = tf.image.random_hue(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

        image = tf.clip_by_value(image, 0.0, 255.0)
    else:
        image = tf.clip_by_value(image, 0.0, 255.0)
    #image = image - tf.reshape(tf.constant([123, 117, 104], dtype=tf.float32), [1, 1, 3])
    #image = tf.concat([image[:, :, 2:3], image[:, :, 1:2], image[:, :, 0:1]], axis=2)

    return image


def random_resize(img_gt, is_training, minval=0.5, maxval=2.0):
    if is_training:
        img_gt = tf.cast(img_gt, tf.float32)
        _img = img_gt[:,:,0:-1]
        _gt = img_gt[:,:,-1:]

        _tar_ratio = tf.random_uniform(shape=[], minval=minval, maxval=maxval)
        _tar_size = tf.cast(
            tf.round(
                _tar_ratio*tf.cast(tf.shape(_img)[0:-1], tf.float32)
            ),
            tf.int32
        )

        _img_resized = tf.image.resize_images(_img,
                                              _tar_size,
                                              align_corners=True,
                                              method=tf.image.ResizeMethod.BICUBIC)
        _gt_resized = tf.image.resize_images(_gt,
                                             _tar_size,
                                             align_corners=True,
                                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        resized_img_gt = tf.concat([_img_resized, _gt_resized], axis=-1)
    else:
        img_gt = tf.cast(img_gt, tf.float32)
        resized_img_gt = img_gt

    return resized_img_gt


def random_crop_fit(img_gt, is_training, crop_size=473):
    _img = img_gt[:, :, 0:-1]
    _gt = img_gt[:, :, -1:]
    if is_training:
        img_fit, gt_fit = tf.cond(
            tf.minimum(tf.shape(_img)[0], tf.shape(_img)[1]) < crop_size,
            lambda:
            [_resize_image_on_shorter_edge(_img, crop_size + 30, method=tf.image.ResizeMethod.BICUBIC),
             _resize_image_on_shorter_edge(_gt, crop_size + 30, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)],
            lambda:
            [_img, _gt]
        )

        img_gt_fit = tf.concat([img_fit, gt_fit], axis=-1)
        frame_tube = tf.random_crop(img_gt_fit, [crop_size, crop_size, 4])
        uniform_random = random_ops.random_uniform([], 0, 1.0)
        mirror_cond = math_ops.less(uniform_random, .5)
        result = control_flow_ops.cond(mirror_cond,
                                       lambda: array_ops.reverse(frame_tube, [1]),
                                       lambda: frame_tube)
    else:
        img_fit, gt_fit = tf.cond(
            tf.minimum(tf.shape(_img)[0], tf.shape(_img)[1]) < crop_size,
            lambda:
            [_resize_image_on_shorter_edge(_img, crop_size + 30, method=tf.image.ResizeMethod.BICUBIC),
             _resize_image_on_shorter_edge(_gt, crop_size + 30, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)],
            lambda:
            [_img, _gt]
        )

        img_gt_fit = tf.concat([img_fit, gt_fit], axis=-1)
        frame_tube = tf.random_crop(img_gt_fit, [crop_size, crop_size, 4])
        uniform_random = random_ops.random_uniform([], 0, 1.0)
        mirror_cond = math_ops.less(uniform_random, .5)
        result = control_flow_ops.cond(mirror_cond,
                                       lambda: array_ops.reverse(frame_tube, [1]),
                                       lambda: frame_tube)

    return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    """Construct a queued batch of images and labels.

    Args:
      image: 3-D Tensor of [height, width, 3] of type.float32.
      label: 1-D Tensor of type.int32
      min_queue_examples: int32, minimum number of samples to retain
        in the queue that provides of batches of examples.
      batch_size: Number of images per batch.
      shuffle: boolean indicating whether to use a shuffling queue.

    Returns:
      images: Images. 4D tensor of [batch_size, height, width, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 96
    example_list = [image, label]
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            example_list,
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 300 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            example_list,
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 100 * batch_size)

    return [images, label_batch]


def read_frames_with_preprocess(filename_queue, is_training):
    st = tf.string_split([filename_queue], tf.constant('-', dtype=tf.string))

    img_raw = tf.read_file(st.values[0])
    gt_raw = tf.read_file(st.values[1])
    img = tf.image.decode_jpeg(img_raw, 3)
    gt = tf.image.decode_png(gt_raw, 1)

    img_gt = tf.concat([img, gt], axis=-1)

    img_gt = random_resize(img_gt, is_training=is_training)

    img_gt = random_crop_fit(img_gt, is_training=is_training, crop_size=473)

    img = img_gt[:,:,0:-1]
    gt = img_gt[:,:,-1:]
    img = single_image_preprocess(img, is_training=is_training)

    print("Ground truth minus 1 to 255 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    return img, tf.to_int64(gt-1)


def batch_for_train(filename, batch_size):
    filename_queue = generate_input_string_queue(filename)
    min_queue_example = 40
    example, label = read_frames_with_preprocess(filename_queue.dequeue(), is_training=True)
    return _generate_image_and_label_batch(example, label, min_queue_example, batch_size, shuffle=True)


def batch_for_validation(filename, batch_size):
    filename_queue = generate_input_string_queue(filename)
    min_queue_example = 10
    example, label = read_frames_with_preprocess(filename_queue.dequeue(), is_training=False)
    return _generate_image_and_label_batch(example, label, min_queue_example, batch_size, shuffle=True)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    images, labels = batch_for_validation(filename='C:/Users/v-yizzh/Documents/code/rl-segmentation/datasets/ade20k_validation_list.txt',
                                     batch_size=2)
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = False

    sess = tf.Session(config=config)
    tf.train.start_queue_runners(sess=sess)
    while True:
        image, label = sess.run([images, tf.squeeze(labels, axis=-1)+1])
        for i in range(0, 2):
                im = Image.fromarray(image[i].astype(np.uint8))
                lb = Image.fromarray(label[i].astype(np.uint8), 'L')
                im.save("D:/workspace/yizhou/train/rl-segmentation/temp/img_" + str(i) + ".png")
                lb.save("D:/workspace/yizhou/train/rl-segmentation/temp/gt_" + str(i) + ".png")


if __name__ == '__main__':
    main()