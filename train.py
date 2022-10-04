import datetime
import wandb
import torch
import sys
import os

import pathlib
import albumentations as A
import numpy as np

from pathlib import Path
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from tqdm import tqdm

from settings import *
from evaluate import evaluate
from unet.model import UNet

from utils.dataset import Dataset, DatasetType
from utils.early_stopping import EarlyStopping
from utils.metrics import SegmentationMetrics

# Logging
from utils.logging import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

torch.backends.cudnn.benchmark = True

class UnetTraining:
    def __init__(self, net: UNet):
        assert net is not None
        self.model = net.to(DEVICE, non_blocking=True)
        self.search_files = IS_SEARCHING_FILES
        self.class_weights = torch.tensor([ 1.0, 27745 / 23889, 27745 / 3502 ], dtype=torch.float).to(DEVICE, non_blocking=True)

        self.batch_size = BATCH_SIZE
        self.num_epochs = NUM_EPOCHS
        self.num_workers = NUM_WORKERS
        self.valid_eval_step = VALID_EVAL_STEP
        self.learning_rate = LEARNING_RATE
        self.pin_memory = PIN_MEMORY
        self.saving_checkpoints = SAVING_CHECKPOINT
        self.using_amp = USING_AMP
        self.patch_size = PATCH_SIZE

        self.get_augmentations()
        self.get_loaders()

        self.device = DEVICE
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=0.0005)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, mode='max', min_lr=1e-8, patience=30, cooldown=30, verbose=True)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters())
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=2e-3, steps_per_epoch=len(self.train_loader), epochs=self.num_epochs)
        self.early_stopping = EarlyStopping(patience=30, verbose=True)
        self.class_labels = {0: 'background', 1: 'fire', 2: 'smoke'}

        if LOAD_MODEL:
            self.load_checkpoint(Path('checkpoints'))
            self.model.to(self.device, non_blocking=True)

    def get_augmentations(self):
        self.train_transforms = A.Compose(
            [
                A.Rotate(limit=(0, 10), p=0.5),
                A.HorizontalFlip(p=0.5),
                A.OneOf([
                    A.ElasticTransform(p=0.3),
                    A.GridDistortion(p=0.4),
                    A.OpticalDistortion(distort_limit=1, shift_limit=0.2, p=0.3),
                ], p=0.8),
                A.RandomBrightnessContrast(p=0.8),
                A.OneOf([
                    A.Blur(p=0.3),
                    A.MotionBlur(p=0.5),
                    A.Sharpen(p=0.2),
                ], p=0.8),
                ToTensorV2(),
            ]
        )

        self.val_transforms = A.Compose(
            [
                ToTensorV2(),
            ],
        )

    def get_loaders(self):
        if self.search_files:
            # Full Dataset
            all_imgs = [file for file in os.listdir(pathlib.Path(
                'data', 'imgs')) if not file.startswith('.')]

            # Split Dataset
            val_percent = 0.6
            n_dataset = int(round(val_percent * len(all_imgs)))

            # Load train & validation datasets
            self.train_dataset = Dataset(data_dir='data', images=all_imgs[:n_dataset], type=DatasetType.TRAIN,
                                         is_combined_data=True, patch_size=self.patch_size, transform=self.train_transforms)
            self.train_dataset.shuffle()
            self.val_dataset = Dataset(data_dir='data', images=all_imgs[n_dataset:], type=DatasetType.VALIDATION,
                                       is_combined_data=True, patch_size=self.patch_size, transform=self.val_transforms)

            # Get Loaders
            self.train_loader = DataLoader(self.train_dataset, num_workers=self.num_workers,
                                           batch_size=self.batch_size, pin_memory=self.pin_memory, shuffle=True)
            self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size,
                                         num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=False)
            return

        self.train_dataset = Dataset(data_dir=r'data', img_dir=r'imgs', type=DatasetType.TRAIN,
                                     is_combined_data=True, patch_size=self.patch_size, transform=self.train_transforms)
        self.train_dataset.shuffle()
        self.val_dataset = Dataset(data_dir=r'data', img_dir=r'imgs', type=DatasetType.VALIDATION,
                                   is_combined_data=True, patch_size=self.patch_size, transform=self.val_transforms)

        # Get Loaders
        self.train_loader = DataLoader(self.train_dataset, num_workers=self.num_workers, batch_size=self.batch_size, pin_memory=self.pin_memory, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=False)

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        if not self.saving_checkpoints:
            return

        if isinstance(self.model, torch.nn.DataParallel):
            self.model = self.model.module

        state = {
            'time': str(datetime.datetime.now()),
            'model_state': self.model.state_dict(),
            'model_name': type(self.model).__name__,
            'optimizer_state': self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch
        }

        log.info('[SAVING MODEL]: Model checkpoint saved!')
        torch.save(state, Path('checkpoints', 'checkpoint.pth.tar'))

        if is_best:
            log.info('[SAVING MODEL]: Saving checkpoint of best model!')
            torch.save(state, Path('checkpoints', 'best-checkpoint.pth.tar'))

    def load_checkpoint(self, path: Path):
        log.info('[LOADING MODEL]: Started loading model checkpoint!')
        best_path = Path(path, 'best-checkpoint.pth.tar')

        if best_path.is_file():
            path = best_path
        else:
            path = Path(path, 'checkpoint.pth.tar')

        if not path.is_file():
            return

        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict['model_state'])
        self.optimizer.load_state_dict(state_dict['optimizer_state'])
        self.optimizer.name = state_dict['optimizer_name']
        log.info(
            f"[LOADING MODEL]: Loaded model with stats: epoch ({state_dict['epoch']}), time ({state_dict['time']})")

    def train(self):
        log.info(f'''[TRAINING]:
            Epochs:          {self.num_epochs}
            Batch size:      {self.batch_size}
            Patch size:      {self.patch_size}
            Learning rate:   {self.learning_rate}
            Training size:   {int(len(self.train_dataset))}
            Validation size: {int(len(self.val_dataset))}
            Checkpoints:     {self.saving_checkpoints}
            Device:          {self.device.type}
            Mixed Precision: {self.using_amp}
        ''')

        wandb_log = wandb.init(project='firebot-unet',
                               resume='allow', entity='firebot031')
        wandb_log.config.update(dict(epochs=self.num_epochs, batch_size=self.batch_size, learning_rate=self.learning_rate,
                                save_checkpoint=self.saving_checkpoints, patch_size=self.patch_size, amp=self.using_amp))

        grad_scaler = torch.cuda.amp.GradScaler(enabled=self.using_amp)
        #criterion = torch.nn.BCEWithLogitsLoss()
        criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights, reduction='mean').to(device=self.device, non_blocking=True)
        metric_calculator = SegmentationMetrics(activation='softmax')

        global_step = 0
        last_best_score = float('inf')

        pixel_acc = 0.0
        dice_score = 0.0
        jaccard_index = 0.0

        torch.cuda.empty_cache()
        for epoch in range(self.num_epochs):
            self.model.train()

            epoch_loss = []
            progress_bar = tqdm(total=int(len(self.train_dataset)), desc=f'Epoch {epoch + 1}/{self.num_epochs}', unit='img', position=0)

            for i, batch in enumerate(self.train_loader):
                # Zero Grad
                self.optimizer.zero_grad(set_to_none=True)

                # Get Batch Of Images
                batch_image = batch['image'].to(self.device, non_blocking=True)
                batch_mask = batch['mask'].to(self.device, non_blocking=True)

                # Predict
                with torch.cuda.amp.autocast(enabled=self.using_amp):
                    masks_pred = self.model(batch_image)
                    metrics = metric_calculator(batch_mask, masks_pred)
                    loss = criterion(masks_pred, batch_mask)

                # Scale Gradients
                grad_scaler.scale(loss).backward()
                # grad_scaler.unscale_(self.optimizer)
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 255)

                grad_scaler.step(self.optimizer)
                grad_scaler.update()
                self.scheduler.step()                

                # Show batch progress to terminal
                progress_bar.update(batch_image.shape[0])
                global_step += 1

                # Calculate training metrics
                pixel_acc += metrics['pixel_acc']
                dice_score += metrics['dice_score']
                jaccard_index += metrics['jaccard_index']
                epoch_loss.append(loss)

                # Evaluation of training
                eval_step = (int(len(self.train_dataset)) // (self.valid_eval_step * self.batch_size))
                if eval_step > 0:
                    if global_step % eval_step == 0:
                        val_loss = evaluate(self.model, self.val_loader, self.device, self.class_labels, wandb_log)
                        progress_bar.set_postfix(**{'Loss': torch.mean(torch.tensor(epoch_loss)).item()})

                        wandb_log.log({
                            'Learning Rate': self.optimizer.param_groups[0]['lr'],
                            'Images [training]': wandb.Image(batch_image[0].cpu(), masks={
                                'ground_truth': {
                                    'mask_data': batch_mask[0].cpu().numpy(),
                                    'class_labels': self.class_labels
                                },
                                'prediction': {
                                    'mask_data': masks_pred.argmax(dim=1)[0].cpu().numpy(),
                                    'class_labels': self.class_labels
                                }
                            }
                            ),
                            'Epoch': epoch,
                            'Pixel Accuracy [training]': metrics['pixel_acc'].item(),
                            'IoU Score [training]': metrics['jaccard_index'].item(),
                            'Dice Score [training]': metrics['dice_score'].item(),
                        })

                        if val_loss < last_best_score:
                            self.save_checkpoint(epoch, True)
                            last_best_score = val_loss

            # Update Progress Bar
            mean_loss = torch.mean(torch.tensor(epoch_loss)).item()
            progress_bar.set_postfix(**{'Loss': mean_loss})
            progress_bar.close()

            wandb_log.log({
                'Loss [training]': mean_loss,
                'Epoch': epoch,
            })

            # Saving last modelself.val_dataset.type
            if self.save_checkpoint:
                self.save_checkpoint(epoch, False)

            # Early Stopping
            if self.early_stopping.early_stop:
                self.save_checkpoint(epoch, True)
                log.info(
                    f'[TRAINING]: Early stopping training at epoch {epoch}!')
                break

        # Push average training metrics
        wandb_log.finish()

if __name__ == '__main__':
    net = UNet(n_channels=3, n_classes=NUM_CLASSES)
    training = UnetTraining(net)

    try:
        training.train()        
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    
    logging.info('[TRAINING]: Training finished!')
