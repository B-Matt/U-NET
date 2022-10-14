import os
import sys
import numpy as np
import albumentations.augmentations.functional as F

from PIL import Image
from pathlib import Path
from collections import namedtuple
from skimage.transform import resize

from utils.logging import logging
from utils.rgb import rgb2mask

# Logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)   

data_info_tuple = namedtuple(
    'data_info_tuple',
    'image, mask'
)

# Image Data Class
class ImageData:
    """
        Class that loads image and mask (used for disk caching).
    """
    def __init__(self, info: data_info_tuple, is_combined: bool = False, patch_size: int = 256, is_searching_dirs: bool = False) -> None:
        self.info = info
        self.is_combined = is_combined
        self.patch_size = patch_size

        if is_searching_dirs == True:
            path = Path(info.image, self._get_file_from_dir(info.image))
            self.input_image = Image.open(path)
        else:
            self.input_image = Image.open(info.image)

        if not self.is_combined:
            try: 
                raise ValueError    
            except ValueError: 
                log.error("Combination of multiple masks currently not implement!")   
                sys.exit(1)  
    
        if is_searching_dirs == True:
            self.input_mask = Image.open(Path(info.mask, self._get_file_from_dir(info.mask)))
        else:
            self.input_mask = Image.open(info.mask)

    def _get_file_from_dir(self, dir):
        for topdir, firs, files in os.walk(dir):
            file = sorted(files)[0]
            return file

    def resize_and_pad(self, img, type):
        old_size = img.size
        ratio = float(self.patch_size) / max(old_size)        
        new_size = tuple([int(x * ratio) for x in old_size])
        img = img.resize(new_size, Image.NEAREST if type == 'mask' else Image.BICUBIC)

        new_img = Image.new("RGB", (self.patch_size, self.patch_size))
        new_img.paste(img, (((self.patch_size - new_size[0]) // 2), ((self.patch_size - new_size[1]) // 2)))
        return new_img

    def get_full_image(self):
        #img = self.resize_and_pad(self.input_image, 'image')
        #mask = self.resize_and_pad(self.input_mask, 'mask')

        img_ndarray = np.asarray(self.input_image, dtype=np.uint8)
        mask_ndarray = np.asarray(self.input_mask, dtype=np.uint8)
        #img_ndarray = img_ndarray / 255

        mask_ndarray = rgb2mask(mask_ndarray)        
        return img_ndarray, mask_ndarray

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