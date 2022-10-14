import gc
import tqdm
import enum
import torch
import string
import pathlib

from multiprocessing.pool import ThreadPool
from os.path import splitext
from typing import List
from torch.utils.data import Dataset

from utils.general import data_info_tuple, NUM_THREADS
from utils.image_data import ImageData, ImageDataDir

# Classes
class DatasetType(enum.Enum):
    TRAIN = 'training_dataset'
    VALIDATION = 'validation_dataset'
    TEST = 'test_dataset'

class Dataset(Dataset):
    def __init__(self,
        data_dir: string,
        img_dir: string,
        images: List = None,
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

        self.img_tupels = []
        if self.is_searching_dirs:
            self.img_tupels = self.preload_image_data_dir(data_dir, img_dir, type)
        else:
            self.img_tupels = self.preload_image_data(data_dir)

        # prefix = '[TRAINING]:' if type == DatasetType.TRAIN else '[VALIDATION]:'
        # fcn = self.load_sample
        # results = ThreadPool(NUM_THREADS).imap(fcn, range(len(self.img_tupels)))
        # pbar = tqdm.tqdm(enumerate(results), total=len(self.img_tupels))
        # self.images_data = []

        # for i, x in pbar:
        #     self.images_data.append(x)
        #     pbar.desc = f'{prefix} Caching data'
        # pbar.close()

        # # Memory Managment
        # self.img_tupels.clear()
        # gc.collect()

    def preload_image_data(self, data_dir: string):
        dataset_files: List = []
        for image in self.all_imgs:
            data_info = data_info_tuple(
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
                    pathlib.Path(path, 'Image'),
                    pathlib.Path(path, 'Mask')
                )
                dataset_files.append(data_info)
        return dataset_files

    def load_sample(self, index):
        if self.is_searching_dirs:
            image_data: ImageDataDir = ImageDataDir(self.img_tupels[index], self.is_combined_data, self.patch_size)
        else:
            image_data: ImageData = ImageData(self.img_tupels[index], self.is_combined_data, self.patch_size)
        return image_data.get_sample()

    def __len__(self):
        return len(self.img_tupels)

    def __getitem__(self, index: int):
        img, mask = self.load_sample(index) #self.images_data[index]

        if self.transform is not None:
            augmentation = self.transform(image=img, mask=mask)
            temp_img = augmentation['image']
            temp_mask = augmentation['mask']

        return {
            'image': torch.as_tensor(temp_img.float()),
            'mask': torch.as_tensor(temp_mask.long())
        }
