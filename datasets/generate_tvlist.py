import os
import glob

def generate_train_val_list(dataset):
    if dataset == 'cityspace':
        train_list_file = open('C:/Users/v-yizzh/Documents/code/rl-segmentation/datasets/cityscape_train_list.txt', 'a')
        val_list = open('C:/Users/v-yizzh/Documents/code/rl-segmentation/datasets/cityscape_validation_list.txt', 'a')
        root_path = "D:/data/yizhou/cityspace"
        img_pth = os.path.join(root_path, 'leftImg8bit')
        gt_pth = os.path.join(root_path, 'gtFine')
        img_train_pth = os.path.join(img_pth, 'train')
        img_validation_pth = os.path.join(img_pth, 'val')
        gt_train_pth = os.path.join(gt_pth, 'train')
        gt_validation_pth = os.path.join(gt_pth, 'val')

        for name in glob.glob(img_validation_pth+'/*/*'):
            this_img_pth = name
            this_img_relpth = this_img_pth.replace(img_validation_pth, '')
            this_img_pth = this_img_pth.replace("\\", '/')
            this_img_relpth = this_img_relpth.replace("\\", '/')
            this_gt_relpth = '_'.join(this_img_relpth[1:].split('_')[0:-1]+['gtFine']+['labelTrainIds']) + '.png'
            this_gt_pth = os.path.join(gt_validation_pth, this_gt_relpth)
            this_gt_pth = this_gt_pth.replace("\\", '/')
            pth_str = ' '.join([this_img_pth, this_gt_pth])
            val_list.write(pth_str+'\n')
    elif dataset == 'ade20k':
        train_list_file = open('C:/Users/v-yizzh/Documents/code/rl-segmentation/datasets/ade20k_train_list.txt', 'a')
        val_list_file = open('C:/Users/v-yizzh/Documents/code/rl-segmentation/datasets/ade20k_validation_list.txt', 'a')
        root_path = "D:/data/yizhou/ADEChallengeData2016/ADEChallengeData2016"
        img_pth = os.path.join(root_path, 'images')
        gt_pth = os.path.join(root_path, 'annotations')
        img_train_pth = os.path.join(img_pth, 'training')
        img_validation_pth = os.path.join(img_pth, 'validation')
        gt_train_pth = os.path.join(gt_pth, 'training')
        gt_validation_pth = os.path.join(gt_pth, 'validation')

        for name in glob.glob(img_train_pth + '/*'):
            this_img_pth = name
            this_img_relpth = this_img_pth.replace(img_train_pth, '')
            this_img_pth = this_img_pth.replace("\\", '/')
            this_img_relpth = this_img_relpth.replace("\\", '/')
            this_gt_relpth = this_img_relpth[1:].split('.')[0] + '.png'
            this_gt_pth = os.path.join(gt_train_pth, this_gt_relpth)
            this_gt_pth = this_gt_pth.replace("\\", '/')
            pth_str = ' '.join([this_img_pth, this_gt_pth])
            train_list_file.write(pth_str + '\n')

if __name__ == '__main__':
    generate_train_val_list('ade20k')

