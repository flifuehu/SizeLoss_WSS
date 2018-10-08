#!/usr/env/bin python3.6

import os
from random import random

from PIL import Image, ImageOps
from torch.utils.data import Dataset
import pandas as pd
import cv2
import torch


# Ignore warnings
# import warnings
# warnings.filterwarnings("ignore")


def make_dataset(root_dir, dataset_path):

    dataset = pd.read_csv(dataset_path)

    images = dataset['data'].values
    labels = dataset['mask'].values
    labels_weak = dataset['mask'].values
    labels_class = dataset['label'].values

    items = []
    for it_im, it_gt, it_w, it_c in zip(images, labels, labels_weak, labels_class):
        item = (os.path.join(root_dir, it_im),      # image
                os.path.join(root_dir, it_gt),      # weak mask
                os.path.join(root_dir, it_w),       # weak mask
                it_c)                               # classification label (tool name)
        items.append(item)

    return items


class MedicalImageDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, mode, root_dir, dataset_path, transform=None, mask_transform=None,
                 augment=False, equalize=False, img_size=(256, 256)):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.mode = mode
        self.root_dir = root_dir
        self.dataset_path = dataset_path
        self.transform = transform
        self.mask_transform = mask_transform
        self.imgs = make_dataset(root_dir, dataset_path)
        self.augmentation = augment
        self.equalize = equalize
        self.img_size = img_size

    def __len__(self):
        return len(self.imgs)

    def augment(self, img, mask):
        if random() > 0.5:
            img = ImageOps.flip(img)
            mask = ImageOps.flip(mask)
        if random() > 0.5:
            img = ImageOps.mirror(img)
            mask = ImageOps.mirror(mask)
        if random() > 0.5:
            angle = random() * 90 - 45
            img = img.rotate(angle)
            mask = mask.rotate(angle)
        return img, mask

    def __getitem__(self, index):
        img_path, mask_path, mask_weak_path, label = self.imgs[index]
        # print("{} and {}".format(img_path,mask_path))
        img = Image.open(img_path).convert('L')  # .convert('RGB')
        mask = Image.open(mask_path).convert('L')  # .convert('RGB')
        mask_weak = Image.open(mask_weak_path).convert('L')

        # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # .convert('RGB')
        # mask_weak = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # .convert('RGB')
        # mask = mask_weak.copy()
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, self.img_size)
        # mask = cv2.resize(mask, self.img_size)
        # mask_weak = cv2.resize(mask_weak, self.img_size)
        # mask = mask[None, ...]
        # mask_weak = mask_weak[None, ...]

        #TODO: change this to use padding instead of simply upscaling
        # width: 376.306630 -> 380
        # height: 216.871258 -> 220
        # mean_R: 99.240233
        # mean_G: 70.150994
        # mean_B: 66.731335

        

        # pad with neutral value to match self.size

        # upscaling to match self.img_size
        img = img.resize(self.img_size, Image.ANTIALIAS)
        mask = mask.resize(self.img_size, Image.ANTIALIAS)
        mask_weak = mask_weak.resize(self.img_size, Image.ANTIALIAS)


        if self.equalize:
            img = ImageOps.equalize(img)

        if self.augmentation:
            img, mask = self.augment(img, mask)

        if self.transform:
            img = self.transform(img)
            mask = self.mask_transform(mask)
            mask_weak = self.mask_transform(mask_weak)


        return [img, mask, mask_weak, img_path]
