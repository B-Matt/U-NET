import torch
import wandb

from tqdm import tqdm
from utils.metrics import BinaryMetrics


def evaluate(net, dataloader, device, class_labels, training):
    net.eval()
    num_val_batches = len(dataloader)
    metric_calculator = BinaryMetrics()
    criterion = torch.nn.BCEWithLogitsLoss()

    pixel_accuracy_sum = 0
    dice_sum = 0
    iou_sum = 0
    recall_sum = 0
    global_loss = 0

    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']

        image = image.to(device=device, non_blocking=True)
        mask_true = mask_true.to(device=device, non_blocking=True)
        mask_pred = None

        with torch.no_grad():
            mask_pred = net(image)
            mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()
            pixel_accuracy, dice_score, precision, specificity, recall, jaccard_score = metric_calculator(mask_true, mask_pred)

            pixel_accuracy_sum += pixel_accuracy
            dice_sum += dice_score
            iou_sum += jaccard_score
            recall_sum += recall

            loss = criterion(mask_pred, mask_true)
            global_loss += loss.item()

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
        'Loss [validation]': global_loss / num_val_batches,
        'Pixel Accuracy [validation]': pixel_accuracy_sum / num_val_batches,
        'IoU Score [validation]': iou_sum / num_val_batches,
        'Dice Score [validation]': dice_sum / num_val_batches,
        'Recalls [validation]': recall_sum / num_val_batches,
    })

    net.train()
    
    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return global_loss

    return global_loss / num_val_batches