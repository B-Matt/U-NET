import os
from PIL import Image
import cv2
import torch
import argparse
import logging

import numpy as np
import torch.nn.functional as F

from unet.model import UNet
from utils.rgb import mask2rgb
from utils.dataset import Dataset
from utils.plots import plot_img_and_mask

# Logging
from utils.logging import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Functions
def predict_img(net, full_img, device, out_threshold=0.5):
    net.eval()

    full_img = np.transpose(full_img, (2, 0, 1))
    img = torch.from_numpy(full_img)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1).cpu()
        else:
            probs = torch.sigmoid(output).cpu()

    if net.n_classes == 1:
        return (probs > out_threshold).numpy()
    else:
        return F.one_hot(probs.argmax(dim=1)[0], net.n_classes).permute(2, 0, 1).numpy()

def get_output_filenames(args):
    def _generate_name(fn):
        split = os.path.splitext(fn)
        return f'{split[0]}_OUT{split[1]}'

    return args.output or list(map(_generate_name, args.input))

def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))

if __name__ == '__main__':
    # Argument Parser
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='checkpoint.pth', metavar='FILE', help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true', help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5, help='Minimum probability value to consider a mask pixel white')
    args = parser.parse_args()

    # Define input and output files
    in_files = args.input
    out_files = get_output_filenames(args)

    # Loading UNET model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(n_channels=3, n_classes=3).to(device=device)
    logging.info(f'[PREDICTION]: Loading model {args.model}')
    logging.info(f'[PREDICTION]: Using device {device}')

    state_dict = torch.load(args.model, map_location=device)
    net.load_state_dict(state_dict['model_state'])
    logging.info('[PREDICTION]: Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'[PREDICTION]: Predicting image {filename}...')

        # Load image
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Dataset._resize_and_pad(img, (768, 768), (0, 0, 0))

        # Predict image
        mask = predict_img(net=net, full_img=img, device=device, out_threshold=args.mask_threshold)
        # mask = mask2rgb(mask)
        
        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask)
            result.save(out_filename)
            logging.info(f'[PREDICTION]: Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'PREDICTION]: Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)
