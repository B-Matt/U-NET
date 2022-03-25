from matplotlib import pyplot as plt

from utils.dataset import Dataset


dataset = Dataset(data_dir = r'data', img_dir = r'imgs', patch_size=1024)

for i in range(1):
    img, mask, image_data_tuple = dataset[i]

   
    plt.figure(i + 1)
    plt.axis('off')

    plt.subplot(1, 2, 1)
    plt.imshow(img.permute(1, 2, 0))

    plt.subplot(1, 2, 2)
    plt.imshow(mask.permute(1, 2, 0))
    
plt.show()