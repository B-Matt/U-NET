import os
from pathlib import Path
import sys

from PIL import Image
from scipy import ndimage
from collections import namedtuple
from skimage.transform import resize

import numpy as np

data_info_tuple = namedtuple(
    'data_info_tuple',
    'image, mask'
)

class ImageData:
    def __init__(self, info: data_info_tuple, is_combined: bool, patch_size: int = 256) -> None:
        self.info = info
        self.is_combined = is_combined
        self.patch_size = patch_size
        
        self.input_image = np.array(Image.open(Path(info.image, self._get_file_from_dir(info.image))))
        self.input_mask = np.array(Image.open(Path(info.mask, self._get_file_from_dir(info.mask))).convert('L'), dtype=np.float32)

    def _get_file_from_dir(self, dir):
        for topdir, firs, files in os.walk(dir):
            file = sorted(files)[0]
            return file

    def get_full_image(self):
        img = self.input_image
        mask = self.input_mask

        # Padding
        height = self.input_image.shape[0]
        width = self.input_image.shape[1]

        pad_width = abs(height - width) // 2
        before = pad_width
        after = pad_width

        if pad_width % 2 != 0:
            before = pad_width
            after = pad_width + 1

        if height > width:
            img = np.pad(img, pad_width=[(0, 0), (before, after), (0, 0)], mode='constant')
            mask = np.pad(mask, pad_width=[(0, 0), (before, after)], mode='constant')

        if width > height:
            img = np.pad(img, pad_width=[(before, after), (0, 0), (0, 0)], mode='constant')
            mask = np.pad(mask, pad_width=[(before, after), (0, 0)], mode='constant')

        # Resizing
        img = resize(img, (self.patch_size, self.patch_size))
        mask = resize(mask, (self.patch_size, self.patch_size))
        return img, mask

    def get_patch_image(self):
        """
            Crops parts of image and mask from the dataset so that 
        """
        if self.patch_size % 2 == 0 and self.is_combined == False:
            patch_size = self.patch_size // 2
        else:
            print("Patch size must be even number!")   
            sys.exit(1)

        # Croping Variables
        crop_height = self.input_image.shape[0]
        crop_width = self.input_image.shape[1]
        crop_pos_x, crop_pos_y = ndimage.center_of_mass(self.input_image)

        # Recalculate crop positions based on patch_size
        if crop_pos_x - patch_size < 0:
            crop_pos_x = patch_size

        if crop_pos_x + patch_size > crop_width:
            crop_pos_x = crop_width - patch_size

        if crop_pos_y - patch_size < 0:
            crop_pos_y = patch_size
        
        if crop_pos_y + patch_size > crop_height:
            crop_pos_y = crop_height - patch_size
        
        # Get correct part of image & mask
        mask = self.input_mask[
            crop_pos_y - patch_size:crop_pos_y + patch_size, 
            crop_pos_x - patch_size:crop_pos_x + patch_size
        ]
        img = self.input_image[
            crop_pos_y - patch_size:crop_pos_y + patch_size,
            crop_pos_x - patch_size:crop_pos_x + patch_size
        ]
       
        return img, mask

    def get_sample(self):
        """
            Returns dataset sample (image and mask) based on the is_combined variable.
        """
        if self.is_combined:
            return self.get_full_image()

        return self.get_patch_image()