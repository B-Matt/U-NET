import torch
import numpy as np

from tqdm import tqdm
from utils.metrics import SegmentationMetrics

def evaluate(net, dataloader, device, training):
    net.eval()
    num_val_batches = len(dataloader)

    criterion = torch.nn.CrossEntropyLoss()
    metric_calculator = SegmentationMetrics(activation='softmax')

    pixel_accuracy = []
    dice_score = []
    iou_score = []
    global_loss = []

    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation', position=1, unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']

        image = image.to(device=device, non_blocking=True)
        mask_true = mask_true.to(device=device, non_blocking=True)

        with torch.no_grad():
            mask_pred = net(image)
            metrics = metric_calculator(mask_true, mask_pred)

            pixel_accuracy.append(metrics['pixel_acc'])
            iou_score.append(metrics['jaccard_index'])
            dice_score.append(metrics['dice_score'])

            loss = criterion(mask_pred, mask_true)

            global_loss.append(loss.cpu())
   
    net.train()
    training.log({
        'Loss [validation]': np.mean(global_loss),
        'Pixel Accuracy [validation]': np.mean(pixel_accuracy),
        'IoU Score [validation]': np.mean(iou_score),
        'Dice Score [validation]': np.mean(dice_score),
    })
    return np.mean(global_loss)