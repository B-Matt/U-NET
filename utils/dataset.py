import enum
import joblib
import random
import functools
import string
import pathlib
from os import listdir
from os.path import splitext
from typing import Any, List
import torch
from torch.utils.data import Dataset

from utils.image_data import ImageData, data_info_tuple

# Cachepathlib
mem = joblib.Memory(cachedir='./cache', compress=5, verbose=0)

# Classes
class DatasetType(enum.Enum):
    TRAIN = 'training'
    VALIDATION = 'validation'
    TEST = 'test'

class Dataset(Dataset):
    def __init__(self, data_dir: string, images: List, type: DatasetType = DatasetType.TRAIN, is_combined_data: bool = True, patch_size: int = 128, transform = None) -> None:
        self.all_imgs = images
        self.images_data = self.preload_image_data(data_dir, type)
        self.is_combined_data = is_combined_data
        self.patch_size = patch_size
        self.transform = transform
        self.load_sample = mem.cache(self.load_sample) # Preventing bug when calling joblib.Memory decorator

    @functools.lru_cache(6)
    def preload_image_data(self, data_dir: string, type: DatasetType):
        dataset_files: List = []
        for image in self.all_imgs:
            data_info = data_info_tuple(
                pathlib.Path(data_dir, 'imgs', image),
                pathlib.Path(data_dir, 'masks', f'{splitext(image)[0]}_label.png')
            )
            dataset_files.append(data_info)
        return dataset_files

    @functools.lru_cache(maxsize=6, typed=True)
    def get_data_object(self, data_info: data_info_tuple, is_combined: bool, patch_size: int = 128):
        return ImageData(data_info, is_combined, patch_size)

    def load_sample(self, image_data_tuple):
        image_data: ImageData = self.get_data_object(image_data_tuple, self.is_combined_data, self.patch_size)
        img, mask = image_data.get_sample()
        return img, mask

    def shuffle(self):
        random.shuffle(self.images_data)

    def __len__(self):
        return len(self.images_data)

    def __getitem__(self, index: Any):
        image_data_tuple = self.images_data[index]
        img, mask = self.load_sample(image_data_tuple)

        if self.transform is not None:
            augmentation = self.transform(image=img, mask=mask)
            temp_img = augmentation['image']
            temp_mask = augmentation['mask']

        temp_mask = temp_mask.unsqueeze(0)
        temp_mask = (temp_mask > 0.5)

        return {
            'image': torch.as_tensor(temp_img, dtype=torch.float32).contiguous(),
            'mask': torch.as_tensor(temp_mask, dtype=torch.float32).contiguous()
        }
