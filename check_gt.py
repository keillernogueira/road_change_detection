import imageio
import numpy as np
from skimage.morphology import dilation, square, disk

img = imageio.imread("C:\\Users\\keill\\Desktop\\Datasets\\road_detection\\area4_mask.tif")
print(img.shape)
print(np.bincount(img.astype(int).flatten()))

print('----------------------')
dil_out = dilation(img, disk(3))
print(img.shape)
print(np.bincount(dil_out.astype(int).flatten()))

imageio.imwrite("C:\\Users\\keill\\Desktop\\Datasets\\road_detection\\area4_mask_dilated.tif", dil_out)
