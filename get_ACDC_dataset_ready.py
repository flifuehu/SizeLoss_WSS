import os
import re
import nibabel as nib
import glob
import cv2
from natsort import natsorted
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def erode_mask(mask, kernel_size):
    # mask is empty
    if mask.sum() == 0:
        return mask
    # if its not empty, erode but don't delete completely
    mask_eroded = np.zeros(mask.shape)
    c = 0
    while mask_eroded.sum() < min_scribble_size:
        kernel_size -= c
        kernel = np.ones((kernel_size,kernel_size), np.uint8)
        mask_eroded = cv2.erode(mask, kernel=kernel)
        c += 1
    return mask_eroded

min_scribble_size = 1  # 1 pixels
source_path = '/home/felix/Downloads/training'
train_dir = '/home/felix/Projects/tool-detection/SizeLoss_WSS/ACDC-2D-All/train/'
val_dir = '/home/felix/Projects/tool-detection/SizeLoss_WSS/ACDC-2D-All/val/'

train_filenames = glob.glob('/home/felix/Projects/tool-detection/SizeLoss_WSS/ACDC-2D-All/train/WeaklyAnnotations/*.png')
val_filenames = glob.glob('/home/felix/Projects/tool-detection/SizeLoss_WSS/ACDC-2D-All/val/WeaklyAnnotations/*.png')
train_filenames = natsorted(train_filenames)
val_filenames = natsorted(val_filenames)

filenames = glob.glob(os.path.join(source_path, '**', '*frame*.nii.gz'))
filenames = natsorted(filenames)

for filename in tqdm(filenames):
    img_obj = nib.load(filename)
    img = img_obj.get_fdata()

    patient_n = re.search('/patient([0-9]{3})/', filename).groups(1)[0]
    frame_n = re.search('frame([0-9]{2})', filename) .groups(1)[0]

    n_slices = img.shape[2]
    for ns in range(n_slices):

        # grab the image
        img_ns = img[:, :, ns]
        print(img_ns.shape)
        continue

        img_ns_res = cv2.resize(img_ns, (256, 256))

        # find out if its train or val, img or gt -> if its gt, save weak labels too
        target_filename = 'patient%s_%s_%d.png' % (patient_n, frame_n, ns+1)

        # Img or GT
        img_or_gt = 'Img'
        if 'gt' in filename:
            img_or_gt = 'GT'
            # keep only label 3
            if(len(np.unique(img_ns_res)) == 3):
                img_ns_res = img_ns_res.astype(np.uint8)
                _, img_ns_res = cv2.threshold(img_ns_res, 2.5, 3, cv2.THRESH_BINARY)
            img_gt_weak = erode_mask(img_ns_res, 10)
        else:
            # if its Img, normalize -> [0,1]
            img_ns_res = (img_ns_res - img_ns_res.min()) / (img_ns_res.max() - img_ns_res.min()) * 255.0

        # train or val
        train_or_val = 'train'
        if any([target_filename in f for f in val_filenames]):
            train_or_val = 'val'

        # save the slice
        target_path = '/home/felix/Projects/tool-detection/SizeLoss_WSS/ACDC-2D-All/%s/%s/%s' % \
                      (train_or_val, img_or_gt, target_filename)
        cv2.imwrite(target_path, img_ns_res)