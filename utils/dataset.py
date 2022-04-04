import enum
import functools
import string
import pathlib
from typing import Any, List
import torch
from torch.utils.data import Dataset

from utils.disk import getCache
from utils.image_data import ImageData, data_info_tuple

# Cache
cache = getCache('DataCache')

# Classes
class DatasetType(enum.Enum):
    TRAIN = 'training'
    VALIDATION = 'validation'
    TEST = 'test'

class Dataset(Dataset):
    def __init__(self, data_dir: string, img_dir: string, type: DatasetType = DatasetType.TRAIN, is_combined_data: bool = True, patch_size: int = 128, transform = None) -> None:
        self.images_data = self.preload_image_data(data_dir, img_dir, type)
        self.is_combined_data = is_combined_data
        self.patch_size = patch_size
        self.transform = transform

    @functools.lru_cache(1)
    def preload_image_data(self, data_dir: string, img_dir: string, type: DatasetType):
        dataset_files: List = []
        with open(pathlib.Path(data_dir, f'{type.value}.txt'), mode='r', encoding='utf-8') as file:
            for line in file:
                path = pathlib.Path(data_dir, img_dir, line.strip())
                data_info = data_info_tuple(
                    pathlib.Path(path, 'Image'),
                    pathlib.Path(path, 'Mask')
                )
                dataset_files.append(data_info)

        return dataset_files

    @functools.lru_cache(maxsize=1, typed=True)
    def get_data_object(self, data_info: data_info_tuple, is_combined: bool, patch_size: int = 128):
        return ImageData(data_info, is_combined, patch_size)

    @cache.memoize(typed=True)
    def load_sample(self, image_data_tuple):
        image_data: ImageData = self.get_data_object(image_data_tuple, self.is_combined_data, self.patch_size)
        img, mask = image_data.get_sample()
        return img, mask

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
            'image': torch.as_tensor(temp_img).float().contiguous(),
            'mask': torch.as_tensor(temp_mask).float().contiguous()
        }
