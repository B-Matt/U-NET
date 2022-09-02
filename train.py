import datetime
import wandb
import torch
import os

import numpy as np
from copy import copy
import albumentations as A

from pathlib import Path
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from tqdm import tqdm

import pathlib
from os import listdir
from os.path import splitext

from evaluate import evaluate
from unet.model import UNet

from utils.dataset import Dataset, DatasetType
from utils.early_stopping import EarlyStopping
from utils.metrics import BinaryMetrics

# Logging
from utils.logging import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# PyTorch Settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# Hyperparameters etc.
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 25
NUM_EPOCHS = 300
NUM_WORKERS = 4
PATCH_SIZE = 256
PIN_MEMORY = True
LOAD_MODEL = False
VALID_EVAL_STEP = 1
SAVING_CHECKPOINT = True
USING_AMP = True

class UnetTraining:
    def __init__(self, net: UNet, batch_size: int, device: torch.device, num_epochs: int, num_workers: int, valid_eval_step : int, learning_rate: float, pin_memory: bool, load_model: bool, saving_checkpoints: bool, using_amp: bool, patch_size: int):
        assert net is not None
        self.model = net.to(device, non_blocking=True)
                       
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_workers = num_workers
        self.valid_eval_step  = valid_eval_step
        self.learning_rate = learning_rate
        self.pin_memory = pin_memory
        self.saving_checkpoints = saving_checkpoints
        self.using_amp = using_amp
        self.patch_size = patch_size

        self.get_augmentations()
        self.get_loaders()

        self.device = device
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, mode='min', min_lr=1e-8, patience=25, cooldown=15, verbose=True)
        self.early_stopping = EarlyStopping(patience=30, verbose=True)
        self.class_labels = { 0: 'background', 1: 'fire' }

        if load_model:
            self.load_checkpoint(Path('checkpoints'))
            self.model.to(self.device, non_blocking=True)

    def get_augmentations(self):
        self.train_transforms = A.Compose(
            [
                A.Rotate(limit=(0, 10), p=0.5),
                A.HorizontalFlip(p=0.6),
                A.VerticalFlip(p=0.4),
                A.GridDistortion(p=0.4),
                A.RandomBrightnessContrast(p=0.8),
                A.RandomGamma(p=0.8),
                A.OneOf([
                    A.Blur(p=0.3),
                    A.MotionBlur(p=0.5),
                    A.Sharpen(p=0.2),
                ], p=0.8),
                A.ToFloat(max_value=255.0),
                ToTensorV2(),
            ]
        )

        self.val_transforms = A.Compose(
        [
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ],
    )

    def get_loaders(self):
        # Full Dataset
        all_imgs = [file for file in listdir(pathlib.Path('data', 'imgs')) if not file.startswith('.')]

        # Split Dataset
        val_percent = 0.6
        n_dataset = int(round(val_percent * len(all_imgs)))

        # Load train & validation datasets
        self.train_dataset = Dataset(data_dir='data', images=all_imgs[:n_dataset], type=DatasetType.TRAIN, is_combined_data=True, patch_size=self.patch_size, transform=self.train_transforms)
        self.val_dataset = Dataset(data_dir='data', images=all_imgs[n_dataset:], type=DatasetType.VALIDATION, is_combined_data=True, patch_size=self.patch_size, transform=self.val_transforms) 

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
        log.info(f"[LOADING MODEL]: Loaded model with stats: epoch ({state_dict['epoch']}), time ({state_dict['time']})")

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

        wandb_log = wandb.init(project='ciirc-unet', resume='allow', entity='firebot031')
        wandb_log.config.update(dict(epochs=self.num_epochs, batch_size=self.batch_size, learning_rate=self.learning_rate, save_checkpoint=self.saving_checkpoints, patch_size=self.patch_size, amp=self.using_amp))

        grad_scaler = torch.cuda.amp.GradScaler(enabled=self.using_amp)
        criterion = torch.nn.BCEWithLogitsLoss()
        metric_calculator = BinaryMetrics()

        global_step = 0
        last_best_score = float('inf')
        training_metrics = {
            'pixel_acc': [],
            'dice_score': [],
            'jaccard_index': []
        }

        for epoch in range(self.num_epochs):
            wandb.watch(self.model)
            self.model.train()

            epoch_loss = []
            progress_bar = tqdm(total=int(len(self.train_dataset)), desc=f'Epoch {epoch + 1}/{self.num_epochs}', unit='img', position=0)

            for batch in self.train_loader:
                # Zero Grad
                for param in self.model.parameters():
                    param.grad = None

                # Get Batch Of Images
                batch_image = batch['image'].to(self.device, non_blocking=True)
                batch_mask = batch['mask'].to(self.device, non_blocking=True)
                metrics = {}

                # Predict
                with torch.cuda.amp.autocast(enabled=self.using_amp):
                    masks_pred = self.model(batch_image)
                    metrics = metric_calculator(batch_mask, masks_pred)
                    loss = criterion(masks_pred, batch_mask)

                # Scale Gradients
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 512)
                
                grad_scaler.step(self.optimizer)
                grad_scaler.update()

                # Show batch progress to terminal
                progress_bar.update(batch_image.shape[0])
                progress_bar.set_postfix(**{'Loss': loss.item() })
                global_step += 1

                # Calculate training metrics
                training_metrics['pixel_acc'].append(metrics['pixel_acc'])
                training_metrics['dice_score'].append(metrics['dice_score'])
                training_metrics['jaccard_index'].append(metrics['jaccard_index'])
                epoch_loss.append(loss)
                
                # Evaluation of training
                eval_step = (int(len(self.train_dataset)) // (self.valid_eval_step * self.batch_size))
                if eval_step > 0:
                    if global_step % eval_step == 0:
                        val_loss = evaluate(self.model, self.val_loader, self.device, self.class_labels, wandb_log)
                        masks_pred = (masks_pred > 0.5).float()
                        self.scheduler.step(val_loss)

                        wandb_log.log({
                            'Learning Rate': self.optimizer.param_groups[0]['lr'],
                            'Images [training]': wandb.Image(batch_image[0].cpu(), masks={
                                    'prediction': {
                                        'mask_data': masks_pred[0].cpu().detach().squeeze(0).numpy(),
                                        'class_labels': self.class_labels
                                    },
                                    'ground_truth': {
                                        'mask_data': batch_mask[0].cpu().detach().squeeze(0).numpy(),
                                        'class_labels': self.class_labels
                                    },
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

            progress_bar.close()
            wandb_log.log({
                'Loss [training]': torch.mean(torch.tensor(epoch_loss)).item(),
                'Epoch': epoch,
            })

            # Saving last modelself.val_dataset.type
            if self.save_checkpoint:
                self.save_checkpoint(epoch, False)

            # Early Stopping
            if self.early_stopping.early_stop:
                self.save_checkpoint(epoch, True)
                log.info(f'[TRAINING]: Early stopping training at epoch {epoch}!')
                break

        # Push average training metrics
        wandb_log.log({
            'Learning Rate': self.optimizer.param_groups[0]['lr'],
            'Epoch': self.num_epochs,
            'Pixel Accuracy [training]': torch.mean(torch.tensor(training_metrics['pixel_acc'])).item(),
            'IoU Score [training]': torch.mean(torch.tensor(training_metrics['jaccard_index'])).item(),
            'Dice Score [training]': torch.mean(torch.tensor(training_metrics['dice_score'])).item(),
        })
        wandb_log.finish()

if __name__ == '__main__':
    net = UNet(n_channels=3, n_classes=1, bilinear=False)
    training = UnetTraining(net, BATCH_SIZE, DEVICE, NUM_EPOCHS, NUM_WORKERS, VALID_EVAL_STEP, LEARNING_RATE, PIN_MEMORY, LOAD_MODEL, SAVING_CHECKPOINT, USING_AMP, PATCH_SIZE)

    training.train()
    logging.info('[TRAINING]: Training finished!')