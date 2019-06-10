from __future__ import print_function
import colorsys
import numpy as np
#from keras.models import Model
from model_util.pspnet.cityscapes_labels import trainId2label
from model_util.pspnet.ade20k_labels import ade20k_id2label
from model_util.pspnet.pascal_voc_labels import voc_id2label


def class_image_to_image(class_id_image, class_id_to_rgb_map):
    """Map the class image to a rgb-color image."""
    colored_image = np.zeros((class_id_image.shape[0], class_id_image.shape[1], 3), np.uint8)
    for i in range(-1,256):
        try:
            if i== -1:
                colored_image[class_id_image[:, :] == i] = (255, 255, 255)
            else:
                cl = class_id_to_rgb_map[i]
                colored_image[class_id_image[:,:]==i] = cl.color
        except KeyError as key_error:
            pass
    return colored_image


def color_class_image(class_image, model_name):
    """Color classed depending on the model_util used."""
    if 'cityscapes' in model_name:
        colored_image = class_image_to_image(class_image, trainId2label)
    elif 'voc' in model_name:
        colored_image = class_image_to_image(class_image, voc_id2label)
    elif 'ade20k' in model_name:
        colored_image = class_image_to_image(class_image, ade20k_id2label)
    else:
        colored_image = add_color(class_image)
    return colored_image


def add_color(img):
    """Color classes a good distance away from each other."""
    h, w = img.shape
    img_color = np.zeros((h, w, 3))
    for i in range(1, 151):
        img_color[img == i] = to_color(i)
    return img_color * 255  # is [0.0-1.0]  should be [0-255]


def to_color(category):
    """Map each category color a good distance away from each other on the HSV color space."""
    v = (category-1)*(137.5/360)
    return colorsys.hsv_to_rgb(v, 1, 1)

'''
def debug(model_util, data):
    """Debug model_util by printing the activations in each layer."""
    names = [layer.name for layer in model_util.layers]
    for name in names[:]:
        print_activation(model_util, name, data)


def print_activation(model_util, layer_name, data):
    """Print the activations in each layer."""
    intermediate_layer_model = Model(inputs=model_util.input,
                                     outputs=model_util.get_layer(layer_name).output)
    io = intermediate_layer_model.predict(data)
    print(layer_name, array_to_str(io))
'''

def array_to_str(a):
    return "{} {} {} {} {}".format(a.dtype, a.shape, np.min(a),
                                   np.max(a), np.mean(a))
