import pandas as pd
import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches


def save_bounding_boxes_GT(dataset_path, target_dir, header_names):

    dataset = pd.read_csv(dataset_path, names=header_names)
    dataset_images = dataset.data.unique()

    n_imgs = len(dataset_images)
    for ni in range(n_imgs):

        img_path = os.path.join(root_dir, dataset_images[ni])
        img_gt_path = img_path.replace('frames_fps_1', 'locgauss_fps_1')

        bbs = dataset[dataset.data == dataset_images[ni]]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_gt = cv2.imread(img_gt_path, cv2.IMREAD_GRAYSCALE)

        for _, bb in bbs.iterrows():
            img_bb = img[int(bb.x_min):int(bb.x_max), int(bb.y_min):int(bb.y_max), :]
            img_bb_gt = img_gt[int(bb.x_min):int(bb.x_max), int(bb.y_min):int(bb.y_max)]
            _, img_bb_gt_th = cv2.threshold(img_bb_gt, bin_th, 255, cv2.THRESH_BINARY)





    


bin_th = 235
root_dir = '/mnt/mlnasrw/Data/LapBypass/Sanjay/'
target_dir = '/mnt/mlnasrw/Data/LapBypass/Sanjay/'

dataset_train_path = '/home/felix/Projects/tool-detection/pytorch-retinanet/data/' \
                     'bypass_cleaned_reduced_margin20px/' \
                     'ds_bypass_tools_segs_to_bb_margin20px_retinanet_train.csv'
dataset_test_path = '/home/felix/Projects/tool-detection/pytorch-retinanet/data/' \
                     'bypass_cleaned_reduced_margin20px/' \
                     'ds_bypass_tools_segs_to_bb_margin20px_retinanet_test.csv'

header_names = ['data', 'x_min', 'y_min', 'x_max', 'y_max', 'label', 'fold', 'img_h', 'img_w']

save_bounding_boxes_GT(dataset_train_path, target_dir, header_names)
save_bounding_boxes_GT(dataset_test_path, target_dir, header_names)