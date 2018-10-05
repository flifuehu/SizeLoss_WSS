import pandas as pd
import glob
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from tqdm import tqdm
from functools import partial


def create_segments_bbs(img_path, dataset):
    img_fullpath = os.path.join(root_dir, img_path)
    img_gt_path = img_fullpath.replace('frames_fps_1', 'locgauss_fps_1')

    bbs = dataset[dataset.data == img_path]
    bbs.reset_index(inplace=True, drop=True)
    img = cv2.imread(img_fullpath, cv2.IMREAD_COLOR)
    img_gt = cv2.imread(img_gt_path, cv2.IMREAD_GRAYSCALE)

    rows = []
    for idx, bb in bbs.iterrows():
        img_bb = img[int(bb.y_min):int(bb.y_max), int(bb.x_min):int(bb.x_max), :]
        img_bb_gt = img_gt[int(bb.y_min):int(bb.y_max), int(bb.x_min):int(bb.x_max)]
        _, img_bb_gt_th = cv2.threshold(img_bb_gt, bin_th, 255, cv2.THRESH_BINARY)

        if visualize:
            # overlay = cv2.addWeighted(img_bb_gt_th, 0.5, img_bb_gt, 0.5, 0)
            # plt.imshow(overlay)
            # plt.show()
            overlay = cv2.addWeighted(img_bb_gt_th, 0.5, img_bb[:, :, 0], 0.5, 0)
            plt.imshow(overlay)
            plt.show()

        img_name = img_fullpath.split(os.sep)[-1]
        img_bb_path = os.sep.join(img_fullpath.split(os.sep)[0:-1])
        img_bb_path = img_bb_path.replace('frames_fps_1', 'bbs_fps_1')
        img_bb_gt_path = os.sep.join(img_fullpath.split(os.sep)[0:-1])
        img_bb_gt_path = img_bb_gt_path.replace('frames_fps_1', 'bbsgt_fps_1')
        if not os.path.exists(img_bb_path):
            try:
                os.makedirs(img_bb_path)
            except FileExistsError:
                print('Folder exists: ', img_bb_path)
            except:
                print('Error occurred: ', sys.exc_info()[0])
                raise
        if not os.path.exists(img_bb_gt_path):
            try:
                os.makedirs(img_bb_gt_path)
            except FileExistsError:
                print('Folder exists: ', img_bb_gt_path)
            except:
                print('Error occurred: ', sys.exc_info()[0])
                raise

        img_name = img_name.replace('.png', '_%d.png' % idx)
        cv2.imwrite(os.path.join(img_bb_path, img_name), img_bb)
        cv2.imwrite(os.path.join(img_bb_gt_path, img_name), img_bb_gt_th)

        row = pd.DataFrame(data={'data': [os.path.join(img_bb_path.replace(root_dir, ''), img_name)],
                                 'mask': [os.path.join(img_bb_gt_path.replace(root_dir, ''), img_name)],
                                 'label': [bb.label],
                                 'fold': [bb.fold],
                                 'img_h': [img_bb.shape[0]],
                                 'img_w': [img_bb.shape[1]]})
        rows.append(row)

    rows = pd.concat(rows)
    rows.reset_index(inplace=True, drop=True)
    return rows


def save_bounding_boxes_GT(dataset_path, header_names):

    dataset = pd.read_csv(dataset_path, names=header_names)
    dataset_images = dataset.data.unique()

    # n_imgs = len(dataset_images)
    # for ni in tqdm(range(n_imgs)):
    #     create_segments_bbs(dataset_images[ni], dataset)

    import multiprocessing

    # no progress bar
    # pool = multiprocessing.Pool(5)
    # pool.map(partial(create_segments_bbs, dataset=dataset), dataset_images)

    # progress bar: method 1
    with multiprocessing.Pool(N_PROC) as p:
        ds = list(tqdm(p.imap(partial(create_segments_bbs, dataset=dataset), dataset_images),
                       total=len(dataset_images)))

    return ds

    # progress bar: method 2
    # with multiprocessing.Pool(N_PROC) as p:
    #     for _ in tqdm(p.imap(partial(create_segments_bbs, dataset=dataset), dataset_images), total=len(dataset_images)):
    #         pass




if __name__ == '__main__':

    N_PROC = 6
    visualize = False
    bin_th = 235
    root_dir = '/mnt/mlnasrw/Data/LapBypass/Sanjay/'

    dataset_train_path = '/home/felix/Projects/tool-detection/pytorch-retinanet/data/' \
                         'bypass_cleaned_reduced_margin20px/' \
                         'ds_bypass_tools_segs_to_bb_margin20px_retinanet_train.csv'
    dataset_test_path = '/home/felix/Projects/tool-detection/pytorch-retinanet/data/' \
                         'bypass_cleaned_reduced_margin20px/' \
                         'ds_bypass_tools_segs_to_bb_margin20px_retinanet_test.csv'

    header_names = ['data', 'x_min', 'y_min', 'x_max', 'y_max', 'label', 'fold', 'img_h', 'img_w']

    ds_test = save_bounding_boxes_GT(dataset_test_path, header_names)
    ds_train = save_bounding_boxes_GT(dataset_train_path, header_names)

    ds_test = pd.concat(ds_test)
    ds_test.sort_values(by='data', inplace=True)
    ds_test.reset_index(inplace=True, drop=True)
    ds_test.to_csv(dataset_test_path.replace('.csv', '_bbs.csv'), index=False)

    ds_train = pd.concat(ds_train)
    ds_train.sort_values(by='data', inplace=True)
    ds_train.reset_index(inplace=True, drop=True)
    ds_train.to_csv(dataset_train_path.replace('.csv', '_bbs.csv'), index=False)
