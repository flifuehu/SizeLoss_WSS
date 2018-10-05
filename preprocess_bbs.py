import pandas as pd
import os
import numpy as np
import cv2
# import matplotlib.pyplot as plt
from tqdm import tqdm

import multiprocessing
from functools import partial


min_mask_area = 50
min_img_size = 50*50
max_avg_h_mult = 1.5
max_avg_w_mult = 1.5

N_PROC = 6
show_masks = True
root_dir = '/mnt/mlnasrw/Data/LapBypass/Sanjay/'

ds_train_path = '/home/felix/Projects/tool-detection/pytorch-retinanet/data/' \
                'bypass_cleaned_reduced_margin20px/' \
                'ds_bypass_tools_segs_to_bb_margin20px_retinanet_train_bbs.csv'
ds_val_path = '/home/felix/Projects/tool-detection/pytorch-retinanet/data/' \
               'bypass_cleaned_reduced_margin20px/' \
               'ds_bypass_tools_segs_to_bb_margin20px_retinanet_test_bbs.csv'


def get_mask_properties(row):
    idx, ds = row
    img_path = ds['data']
    mask_path = ds['mask']
    label = ds['label']
    fold = ds['fold']

    img = cv2.imread(os.path.join(root_dir, img_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(os.path.join(root_dir, mask_path), cv2.IMREAD_GRAYSCALE)
    img_mean_RGB = img.mean(axis=(0, 1))
    area = np.sum(mask >= 235)
    img_h, img_w = mask.shape
    ratio_mask_img = area / (img_h*img_w)
    props = pd.DataFrame(data={'mask_path': [mask_path],
                               'mask_area': [area],
                               'img_path': [img_path],
                               'img_mean_R': img_mean_RGB[2],
                               'img_mean_G': img_mean_RGB[1],
                               'img_mean_B': img_mean_RGB[0],
                               'img_h': [img_h],
                               'img_w': [img_w],
                               'img_size': [img_h*img_w],
                               'ratio_mask_img': [ratio_mask_img],
                               'label': [label],
                               'fold': [fold]})
    # if show_masks:
    #     plt.imshow(mask)
    #     plt.show()

    return props

def preprocess_image(row, t):

    idx, data = row

    return row


if __name__ == '__main__':

    ds = pd.read_csv(ds_train_path)
    with multiprocessing.Pool(N_PROC) as p:
        rows = list(tqdm(p.imap(get_mask_properties, ds.iterrows()), total=len(ds)))

    ds_stats = pd.concat(rows)
    ds_stats.sort_values(by='mask_path', inplace=True)
    ds_stats.reset_index(inplace=True, drop=True)
    ds_stats.to_csv('ds_stats.csv', index=False)

    ds_stats_s = ds_stats.sort_values(by='mask_area')
    ds_stats_s.to_csv('ds_stats_sorted_mask_area.csv', index=False)
    ds_stats_r = ds_stats.sort_values(by='ratio_mask_img')
    ds_stats_r.to_csv('ds_stats_sorted_ratio_mask_img.csv', index=False)

    mean_R = ds_stats.img_mean_R.mean()
    mean_G = ds_stats.img_mean_G.mean()
    mean_B = ds_stats.img_mean_B.mean()

    mask_area_min = ds_stats.mask_area.min()
    mask_area_max = ds_stats.mask_area.max()
    mask_area_avg = ds_stats.mask_area.mean()

    img_h_min = ds_stats.img_h.min()
    img_h_max = ds_stats.img_h.max()
    img_h_avg = ds_stats.img_h.mean()

    img_w_min = ds_stats.img_w.min()
    img_w_max = ds_stats.img_w.max()
    img_w_avg = ds_stats.img_w.mean()

    ratio_mask_img_min = ds_stats.ratio_mask_img.min()
    ratio_mask_img_max = ds_stats.ratio_mask_img.max()
    ratio_mask_img_avg = ds_stats.ratio_mask_img.mean()

    # clean the dataset:
    # - remove images with mask area < min_mask_area
    # - remove images with img_size < min_img_size
    # - remove images with img_h < max_avg_h_mult * img_h_avg
    # - remove images with img_w < max_avg_w_mult * img_w_avg
    print('Cleaning dataset...')
    ds_clean = ds_stats[(ds_stats.mask_area > min_mask_area) &
                        (ds_stats.img_size > min_img_size) &
                        (ds_stats.img_h < max_avg_h_mult*img_h_avg) &
                        (ds_stats.img_w < max_avg_w_mult*img_w_avg)]

    transforms = {'width': max_avg_w_mult*img_w_avg,
                  'height': max_avg_h_mult*img_h_avg,
                  'mean_R': mean_R,
                  'mean_G': mean_G,
                  'mean_B': mean_B}

    print('Padding and resizing images...')
    with multiprocessing.Pool(N_PROC) as p:
        processed_img = list(tqdm(p.imap(partial(preprocess_image, t=transforms), ds.iterrows()), total=len(ds.clean)))

    print('Done!')


