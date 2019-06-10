from __future__ import print_function
from __future__ import division

from math import ceil
from scipy import misc, ndimage

import numpy as np
import tensorflow as tf

import model_util.pspnet.utils as utils
import matplotlib.pyplot as plt


def pad_image(img, target_size):
    """Pad an image up to the target size."""
    rows_missing = target_size[0] - img.shape[0]
    cols_missing = target_size[1] - img.shape[1]
    padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, 0)), 'constant')
    return padded_img


def visualize_prediction(prediction):
    """Visualize prediction."""
    cm = np.argmax(prediction, axis=2) + 1
    color_cm = utils.add_color(cm)
    plt.imshow(color_cm)
    plt.show()


def predict_sliding(full_image, net, flip_evaluation):
    """Predict on tiles of exactly the network input shape so nothing gets squeezed."""
    tile_size = net.input_shape
    classes = net.model.outputs[0].shape[3]
    overlap = 1/3

    stride = ceil(tile_size[0] * (1 - overlap))
    tile_rows = int(ceil((full_image.shape[0] - tile_size[0]) / stride) + 1)  # strided convolution formula
    tile_cols = int(ceil((full_image.shape[1] - tile_size[1]) / stride) + 1)
    print("Need %i x %i prediction tiles @ stride %i px" % (tile_cols, tile_rows, stride))
    full_probs = np.zeros((full_image.shape[0], full_image.shape[1], classes))
    count_predictions = np.zeros((full_image.shape[0], full_image.shape[1], classes))
    tile_counter = 0
    for row in range(tile_rows):
        for col in range(tile_cols):
            x1 = int(col * stride)
            y1 = int(row * stride)
            x2 = min(x1 + tile_size[1], full_image.shape[1])
            y2 = min(y1 + tile_size[0], full_image.shape[0])
            x1 = max(int(x2 - tile_size[1]), 0)  # for portrait images the x1 underflows sometimes
            y1 = max(int(y2 - tile_size[0]), 0)  # for very few rows y1 underflows

            img = full_image[y1:y2, x1:x2]
            padded_img = pad_image(img, tile_size)
            # plt.imshow(padded_img)
            # plt.show()
            tile_counter += 1
            print("Predicting tile %i" % tile_counter)
            padded_prediction = net.predict(padded_img, flip_evaluation)
            prediction = padded_prediction[0:img.shape[0], 0:img.shape[1], :]
            count_predictions[y1:y2, x1:x2] += 1
            full_probs[y1:y2, x1:x2] += prediction  # accumulate the predictions also in the overlapping regions

    # average the predictions in the overlapping regions
    full_probs /= count_predictions
    # visualize normalization Weights
    # plt.imshow(np.mean(count_predictions, axis=2))
    # plt.show()
    return full_probs


def predict_multi_scale(full_image, net, scales, sliding_evaluation, flip_evaluation):
    """Predict an image by looking at it with different scales."""
    classes = net.model.outputs[0].shape[3]
    full_probs = np.zeros((full_image.shape[0], full_image.shape[1], classes))
    h_ori, w_ori = full_image.shape[:2]
    for scale in scales:
        print("Predicting image scaled by %f" % scale)
        scaled_img = misc.imresize(full_image, size=scale, interp="bilinear")
        if sliding_evaluation:
            scaled_probs = predict_sliding(scaled_img, net, flip_evaluation)
        else:
            scaled_probs = net.predict(scaled_img, flip_evaluation)
        # scale probs up to full size
        h, w = scaled_probs.shape[:2]
        probs = ndimage.zoom(scaled_probs, (1.*h_ori/h, 1.*w_ori/w, 1.),order=1, prefilter=False)
        # visualize_prediction(probs)
        # integrate probs over all scales
        full_probs += probs
    full_probs /= len(scales)
    return full_probs



if __name__ == "__main__":
    print('None;')
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_util', type=str, default='pspnet101_cityscapes',
                        help='Model/Weights to use',
                        choices=['pspnet50_ade20k',
                                 'pspnet101_cityscapes',
                                 'pspnet101_voc2012'])
    parser.add_argument('-i', '--input_path', type=str, default='example_images/ade20k.jpg',
                        help='Path the input image')
    parser.add_argument('-o', '--output_path', type=str, default='example_results/ade20k.jpg',
                        help='Path to output')
    parser.add_argument('--id', default="1")
    parser.add_argument('-s', '--sliding', action='store_true',
                        help="Whether the network should be slided over the original image for prediction.")
    parser.add_argument('-f', '--flip', action='store_true',
                        help="Whether the network should predict on both image and flipped image.")
    parser.add_argument('-ms', '--multi_scale', action='store_true',
                        help="Whether the network should predict on multiple scales.")
    args = parser.parse_args()

    # environ["CUDA_VISIBLE_DEVICES"] = args.id

    iplhd = inputs=tf.placeholder(dtype=tf.float32, shape=[2, 473, 473, 3])

    if "pspnet50" in args.model_util:
        pspnet = PSPNet(nb_classes=150, resnet_layers=50,
                        inputs=inputs, weights_path=args.model_util, ground_truth=None)
    elif "pspnet101" in args.model_util:
        if "cityscapes" in args.model_util:
            pspnet = PSPNet(nb_classes=19, resnet_layers=101,
                        inputs=inputs, weights_path=args.model_util, ground_truth=None)
        if "voc2012" in args.model_util:
            pspnet = PSPNet(nb_classes=21, resnet_layers=101,
                        inputs=inputs, weights_path=args.model_util, ground_truth=None)

    else:
        print("Network architecture not implemented.")

    logits = pspnet.inference

    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        #sess.run(init_op)
        saver.restore(sess, 'weights/test.ckpt')

        cap = cv2.VideoCapture(args.input_path)
        print(args)
        counter = 0

        if args.multi_scale:
            EVALUATION_SCALES = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]  # must be all floats!
            EVALUATION_SCALES = [0.15, 0.25, 0.5]  # must be all floats!

        time_sum = 0
        while(True):
            # Capture frame-by-frame
            ret, img = cap.read()
            if img is None:
                break

            # img = cv2.resize(img,(int(16.0*713/9.0),713))
            img_orig = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = cv2.resize(img_orig, (473, 473), interpolation=cv2.INTER_AREA)
            img = np.asarray(img, dtype=np.float32)
            img = img[np.newaxis, :, :, :]
            img = np.concatenate((img, img), axis=0)
            start = datetime.datetime.now()
            # class_scores = predict_multi_scale(img, pspnet, EVALUATION_SCALES, args.sliding, args.flip)
            class_scores = sess.run(logits,
                                    feed_dict={
                                        iplhd:img
                                    })
            #class_scores = cv2.resize(class_scores[0, :, :, :], (h_ori, w_ori))
            # End time
            class_scores = class_scores[0, :, :, :]

            full_probs = np.zeros((img_orig.shape[0], img_orig.shape[1], 150))
            h_ori, w_ori = img_orig.shape[:2]
            # scale probs up to full size
            probs = cv2.resize(class_scores, (w_ori, h_ori))
            # visualize_prediction(probs)
            # integrate probs over all scales
            class_scores = full_probs + probs

            end = datetime.datetime.now()

            # Time elapsed
            diff = end - start

            class_image = np.argmax(class_scores, axis=2)
            pm = np.max(class_scores, axis=2)
            colored_class_image = utils.color_class_image(class_image, args.model_util)

            alpha_blended = 0.5 * colored_class_image + 0.5 * img_orig
            filename, ext = splitext(args.output_path)

            time_sum += diff.microseconds/1000.0
            print(counter,diff.microseconds/1000.0,'ms')


            #cv2.putText(alpha_blended,'PSPNet Prediction time: %.0fms (%.1f fps) AVG: %.0fms (%.1f fps)'%(diff.microseconds/1000.0,1000000.0/diff.microseconds,time_sum/(counter+1),1000.0/(time_sum/(counter+1))),(100,100), cv2.FONT_HERSHEY_SIMPLEX, 3,(0,0,0),16,cv2.LINE_AA)
            #cv2.putText(alpha_blended,'PSPNet Prediction time: %.0fms (%.1f fps) AVG: %.0fms (%.1f fps)'%(diff.microseconds/1000.0,1000000.0/diff.microseconds,time_sum/(counter+1),1000.0/(time_sum/(counter+1))),(100,100), cv2.FONT_HERSHEY_SIMPLEX, 3,(255,255,255),10,cv2.LINE_AA)

            misc.imsave(filename + "_%08d_seg"%counter + ext, colored_class_image)
            # misc.imsave(filename + "_%08d_probs"%counter + ext, pm)
            misc.imsave(filename + "_%08d_seg_blended"%counter + ext, alpha_blended)
            counter = counter + 1
    '''