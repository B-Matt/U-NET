import os
import cv2
import tqdm
import enum
import torch
import string
import pathlib
import numpy as np

from pathlib import Path
from multiprocessing.pool import ThreadPool
from os.path import splitext
from typing import List, Tuple
from torch.utils.data import Dataset

from utils.general import data_info_tuple, NUM_THREADS
from utils.image_data import ImageData, ImageDataDir
from utils.rgb import rgb2mask

# Classes
class DatasetType(enum.Enum):
    TRAIN = 'training_dataset'
    VALIDATION = 'validation_dataset'
    TEST = 'test_dataset'

class DatasetCacheType(enum.Enum):
    NONE = 0,
    RAM = 1,
    DISK = 2,

class Dataset(Dataset):
    def __init__(self,
        data_dir: string,
        img_dir: string,
        images: List = None,
        cache_type: DatasetCacheType = DatasetCacheType.NONE,
        type: DatasetType = DatasetType.TRAIN,
        is_combined_data: bool = True,
        patch_size: int = 128,
        transform = None
    ) -> None:
        self.all_imgs = images
        self.is_searching_dirs = images == None and img_dir != None
        self.is_combined_data = is_combined_data
        self.patch_size = patch_size
        self.transform = transform
        self.cache_type = cache_type
        self.images_data = []
        
        self.img_tupels = []
        if self.is_searching_dirs:
            self.img_tupels = self.preload_image_data_dir(data_dir, img_dir, type)
        else:
            self.img_tupels = self.preload_image_data(data_dir)

        if cache_type != DatasetCacheType.NONE:
            prefix = '[TRAINING]:' if type == DatasetType.TRAIN else '[VALIDATION]:'        
            fcn = self.cache_images_to_disk if cache_type == DatasetCacheType.DISK else self.load_sample
            with ThreadPool(NUM_THREADS) as pool:
                results = pool.imap(fcn, range(len(self.img_tupels)))
                pbar = tqdm.tqdm(enumerate(results), total=len(self.img_tupels))

                for i, x in pbar:
                    if cache_type == DatasetCacheType.RAM:
                        self.images_data.append(x)
                    pbar.desc = f'{prefix} Caching data'
                pbar.close()

    def preload_image_data(self, data_dir: string):
        dataset_files: List = []
        for image in self.all_imgs:
            data_info = data_info_tuple(
                image,
                pathlib.Path(data_dir, 'imgs', image),
                pathlib.Path(data_dir, 'masks', f'{splitext(image)[0]}_label.png')
            )
            dataset_files.append(data_info)
        return dataset_files

    def preload_image_data_dir(self, data_dir: string, img_dir: string, type: DatasetType):
        dataset_files: List = []
        with open(pathlib.Path(data_dir, f'{type.value}.txt'), mode='r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                path = pathlib.Path(data_dir, img_dir, line.strip())
                data_info = data_info_tuple(
                    line.strip(),
                    pathlib.Path(path, 'Image'),
                    pathlib.Path(path, 'Mask')
                )
                dataset_files.append(data_info)
        return dataset_files

    def load_sample(self, index):
        info = self.img_tupels[index]
        cache_path = Path('cache', info.name)
        if self.cache_type == DatasetCacheType.DISK and cache_path.exists():
            return np.load(cache_path)
        
        input_image = cv2.imread(str(Path(info.image, os.listdir(info.image)[0])))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        input_mask = cv2.imread(str(Path(info.mask, '0.png')))
        input_mask = cv2.cvtColor(input_mask, cv2.COLOR_BGR2RGB)

        input_image = Dataset._resize_and_pad(input_image, (self.patch_size, self.patch_size), (0, 0, 0))
        input_mask = Dataset._resize_and_pad(input_mask, (self.patch_size, self.patch_size), (0, 0, 0))
        input_mask = rgb2mask(input_mask)
        return input_image, input_mask

    def cache_images_to_disk(self, index):
        info = self.img_tupels[index]
        cache_path = Path('cache', info.name)
       
        if not cache_path.exists():
            img, mask = self.load_sample(index)
            np.savez(cache_path, img, mask)

    @staticmethod
    def _resize_and_pad(image: np.array, 
                    new_shape: Tuple[int, int],
                    padding_color: Tuple[int] = (0, 0, 0)
                ) -> np.array:
        """
        Maintains aspect ratio and resizes with padding.
        Params:
            image: Image to be resized.
            new_shape: Expected (width, height) of new image.
            padding_color: Tuple in BGR of padding color
        Returns:
            image: Resized image with padding
        """
        original_shape = (image.shape[1], image.shape[0])
        ratio = float(max(new_shape)) / max(original_shape)
        new_size = tuple([int(x*ratio) for x in original_shape])
        image = cv2.resize(image, new_size)

        delta_w = new_shape[0] - new_size[0]
        delta_h = new_shape[1] - new_size[1]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)

        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
        return image    

    def __len__(self):
        return len(self.img_tupels)

    def __getitem__(self, index: int):
        if self.cache_type == DatasetCacheType.RAM:
            img, mask = self.images_data[index]
        else:
            img, mask = self.load_sample(index)

        if self.transform is not None:
            augmentation = self.transform(image=img, mask=mask)
            temp_img = augmentation['image']
            temp_mask = augmentation['mask']

        return {
            'image': torch.as_tensor(temp_img.float()),
            'mask': torch.as_tensor(temp_mask.long())
        }
