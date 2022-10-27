import numpy as np

LABEL_COLORS = {
    0: [0, 0, 0],           # Background
    1: [139, 189, 7],       # Fire
    2: [198, 36, 125]       # Smoke
}

def rgb2mask(rgb):
    mask = np.zeros((rgb.shape[0], rgb.shape[1]))
    for k,v in LABEL_COLORS.items():
        mask[np.all(rgb==v, axis=2)] = k
    return mask

def mask2rgb(mask):
    rgb = np.zeros(mask.shape+(3,), dtype=np.uint8)
    for i in np.unique(mask):
        rgb[mask==i] = LABEL_COLORS[i]
    return rgb