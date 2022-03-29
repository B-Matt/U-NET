import datetime
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate import evaluate
from unet.model import UNet
from utils.dice_score import dice_loss
from utils.dataset import Dataset, DatasetType

# Logging
from utils.logging import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
PATCH_SIZE = 128
PIN_MEMORY = True
LOAD_MODEL = False
VALID_EVAL_STEP = 2
SAVING_CHECKPOINT = True
USING_AMP = True

class UnetTraining:
    def __init__(self, net: UNet, batch_size: int, device: torch.device, num_epochs: int, num_workers: int, valid_eval_step : int, learning_rate: float, pin_memory: bool, load_model: bool, saving_checkpoints: bool, using_amp: bool):
        self.net = net
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_workers = num_workers
        self.valid_eval_step  = valid_eval_step 
        self.learning_rate = learning_rate
        self.pin_memory = pin_memory
        self.load_model = load_model
        self.saving_checkpoints = saving_checkpoints
        self.using_amp = using_amp

        self.device = device
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)

        if torch.cuda.is_available():
            self.batch_size *= torch.cuda.device_count()

        self.get_augmentations()
        self.get_loaders()

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
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )

        self.val_transforms = A.Compose(
        [
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    def get_loaders(self):
        self.train_dataset = Dataset(data_dir = r'data', img_dir = r'imgs', type=DatasetType.TRAIN, is_combined_data=True, patch_size=256, transform=self.train_transforms)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=True)

        self.val_dataset = Dataset(data_dir = r'data', img_dir = r'imgs', type=DatasetType.VALIDATION, is_combined_data=True, patch_size=256, transform=self.val_transforms)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=False)

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        model = self.load_model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        state = {
            'time': str(datetime.datetime.now()),
            'model_state': model.state_dict(),
            'model_name': type(model).__name__,
            'optimizer_state': self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch
        }

        log.info('[SAVING MODEL]: Model checkpoint saved!')
        torch.save(state, 'checkpoint.pth.tar')

        if is_best:
            log.info('[SAVING MODEL]: Saving checkpoint of best model!')
            torch.save(state, 'best-checkpoint.pth.tar')

    def load_checkpoint(self, path):
        log.info('[LOADING MODEL]: Started loading model checkpoint!')
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict['model_state'])
        self.optimizer.load_state_dict(state_dict['optimizer_state'])
        self.optimizer.name = state_dict['optimizer_name']
        log.info(f"[LOADING MODEL]: Loaded model with stats: epoch ({state_dict['epoch']}), time ({state_dict['time']})")

    def train(self):
        log.info(f'''[TRAINING]:
            Epochs:          {self.num_epochs}
            Batch size:      {self.batch_size}
            Learning rate:   {self.learning_rate}
            Training size:   {int(len(self.train_dataset))}
            Validation size: {int(len(self.val_dataset))}
            Checkpoints:     {self.saving_checkpoints}
            Device:          {self.device.type}
            Mixed Precision: {self.using_amp}
        ''')

        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=2)
        grad_scaler = torch.cuda.amp.GradScaler(enabled=self.using_amp)
        criterion = torch.nn.CrossEntropyLoss()
        global_step = 0

        last_best_score = 0
        last_best_epoch = 0

        for epoch in range(self.num_epochs):
            net.train()
            epoch_loss = 0
            progress_bar = tqdm(total=int(len(self.train_dataset)), desc=f'Epoch {epoch + 1}/{self.num_epochs}', unit='img')

            for batch in self.train_loader:
                images = batch[0].to(device=self.device, dtype=torch.float32)
                masks = batch[1].to(device=self.device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=self.using_amp):
                    masks_pred = net(images)
                    loss = criterion(masks_pred, masks) + dice_loss(F.softmax(masks_pred, dim=1).float(), F.one_hot(masks, net.n_classes).permute(0, 3, 1, 2).float(),  multiclass=True)

                self.optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(self.optimizer)
                grad_scaler.update()

                progress_bar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                progress_bar.set_postfix(**{'loss (batch)': loss.item()})

                eval_step = (int(len(self.train_dataset)) // (10 * self.batch_size))
                if eval_step > 0:
                    if global_step % eval_step == 0:
                        val_score = evaluate(net, self.val_loader, self.device)
                        scheduler.step(val_score)

                        if val_score > last_best_score:
                            self.save_checkpoint(epoch, True)
                            last_best_score = val_score
                            last_best_epoch = epoch

            if self.save_checkpoint:
                self.save_checkpoint(epoch, False)

            # Early Stopping
            if epoch - last_best_epoch > self.valid_eval_step  * 3:
                self.save_checkpoint(epoch, True)
                log.info(f'[TRAINING]: Early stopping training at epoch ${epoch}!')
                break

if __name__ == '__main__':
    net = UNet(n_channels=3, n_classes=2, bilinear=False).to(DEVICE)
    training = UnetTraining(net, BATCH_SIZE, DEVICE, NUM_EPOCHS, NUM_WORKERS, VALID_EVAL_STEP, LEARNING_RATE, PIN_MEMORY, LOAD_MODEL, SAVING_CHECKPOINT, USING_AMP)
    training.train()
