import torch
import wandb

import torch.nn.functional as F
from tqdm import tqdm

from utils.metrics import BinaryMetrics


def evaluate(net, dataloader, device, training):
    net.eval()
    num_val_batches = len(dataloader)
    metric_calculator = BinaryMetrics()

    pixel_accuracy_sum = 0
    dice_sum = 0
    iou_sum = 0
    recall_sum = 0    

    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']

        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.float32)
        mask_pred = None

        with torch.no_grad():
            mask_pred = net(image)

            if net.n_classes == 1:
                mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()
                pixel_accuracy, dice_score, precision, specificity, recall, jaccard_score = metric_calculator(mask_true, mask_pred)

                pixel_accuracy_sum += pixel_accuracy
                dice_sum += dice_score
                iou_sum += jaccard_score
                recall_sum += recall

            else:
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)

            
    training.log({
        'images (training)': wandb.Image(image[0].cpu()),
        'masks (training)': [ 
            wandb.Image(mask_true[0].float().cpu()),
            wandb.Image(mask_pred[0].cpu()),
        ],
        'Pixel Accuracy (validation)': pixel_accuracy_sum / num_val_batches,
        'IoU Score (validation)': iou_sum / num_val_batches,
        'Dice Score (validation)': dice_sum / num_val_batches,
        'Recalls (validation)': recall_sum / num_val_batches,
    })

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score

    return dice_score / num_val_batches