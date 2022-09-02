import torch
import wandb

import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from utils.metrics import BinaryMetrics


def evaluate(net, dataloader, device, class_labels, training):
    net.eval()
    num_val_batches = len(dataloader)
    metric_calculator = BinaryMetrics()
    criterion = torch.nn.BCEWithLogitsLoss()

    pixel_accuracy = []
    dice_score = []
    iou_score = []
    global_loss = []

    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation', position=1, unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']

        image = image.to(device=device, non_blocking=True)
        mask_true = mask_true.to(device=device, non_blocking=True)
        mask_pred = None

        with torch.no_grad():
            mask_pred = net(image)
            mask_pred = (mask_pred > 0.5).float()
            metrics = metric_calculator(mask_true, mask_pred)

            pixel_accuracy.append(metrics['pixel_acc'])
            dice_score.append(metrics['dice_score'])
            iou_score.append(metrics['jaccard_index'])

            loss = criterion(mask_pred, mask_true)
            global_loss.append(loss.cpu())

    training.log({
        'Images [validation]': wandb.Image(image[0].cpu(), masks={
                'ground_truth': {
                    'mask_data': mask_true[0].cpu().detach().squeeze(0).numpy(),
                    'class_labels': class_labels
                },
                'prediction': {
                    'mask_data': mask_pred[0].cpu().detach().squeeze(0).numpy(),
                    'class_labels': class_labels
                },
            }
        ),
        'Loss [validation]': torch.mean(torch.tensor(global_loss).cpu()).item(),
        'Pixel Accuracy [validation]': torch.mean(torch.tensor(pixel_accuracy).cpu()).item(),
        'IoU Score [validation]': torch.mean(torch.tensor(iou_score).cpu()).item(),
        'Dice Score [validation]': torch.mean(torch.tensor(dice_score).cpu()).item(),
    })

    net.train()
    
    return np.average(torch.tensor(global_loss).cpu())