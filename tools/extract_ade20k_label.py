
import os

import numpy as np

from PIL import Image
import scipy.io as sio

def _extract_segment_label(file):
    img = Image.open(file)
    img_array = np.asarray(img)

    _r= img_array[:,:,0:1]
    _g= img_array[:,:,1:2]
    _b= img_array[:,:,2:3]

    label_mask = (_r.astype(np.uint16)/10)*256+_g.astype(np.uint16)

    return label_mask

def extract_and_store(root_path):
    obj_name = sio.loadmat(os.path.join(root_path, 'index_ade20k.mat'))['index']['objectnames']
    #obj_class = sio.loadmat(os.path.join(root_path, 'index_ade20k.mat'))['objects']['class']
    print(obj_name.shape)
    train_path = os.path.join(root_path, 'images/training')
    validation_path = os.path.join(root_path, 'images/validation')

    for _, sub_1_dir, _ in os.walk(train_path):
        for _, sub_2_dir, _ in os.walk(sub_1_dir):
            for this_root_dir, _, data_file_name in os.walk(sub_2_dir):
                if '.jpg' in data_file_name:
                    data_file_path = os.path.join(this_root_dir, data_file_name)
                    print(data_file_path)

if __name__=='__main__':
    extract_and_store('D:/data/yizhou/ade20k/ADE20K_2016_07_26')