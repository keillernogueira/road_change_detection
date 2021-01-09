import os
import imageio
from skimage import img_as_float
import numpy as np

from utils import check_mkdir


def process_images(input_path, image_list, output_path, crop_size, stride_size, is_train):
    # Loop trough images and create crops
    for im in image_list:
        print("Processing image: " + im)

        # Open image
        img = img_as_float(imageio.imread(os.path.join(input_path, 'area' + im +
                                                       '_landsat8_toa_2013_pansharpen.tif')))
        mask = (imageio.imread(os.path.join(input_path, 'area' + im + '_mask.png'))).astype(int)
        w, h = mask.shape

        crop_count = 0
        for i in range(0, w, stride_size):
            for j in range(0, h, stride_size):
                cur_x = i
                cur_y = j
                patch = img[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :]

                if len(patch) != crop_size and len(patch[0]) != crop_size:
                    cur_x = cur_x - (crop_size - len(patch))
                    cur_y = cur_y - (crop_size - len(patch[0]))
                    patch = img[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]
                elif len(patch) != crop_size:
                    cur_x = cur_x - (crop_size - len(patch))
                    patch = img[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]
                elif len(patch[0]) != crop_size:
                    cur_y = cur_y - (crop_size - len(patch[0]))
                    patch = img[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]

                assert patch.shape == (crop_size, crop_size), "Error create_distributions_over_classes: " \
                                                              "Current patch size is " + str(len(patch)) + \
                                                              "x" + str(len(patch[0]))

                patch_mask = mask[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]

                # we need to check that the patch has road in it
                count = np.bincount(patch_mask.astype(int).flatten())
                if len(count) == 2:  # has road
                    if is_train:
                        # Save crop
                        imageio.imwrite(os.path.join(output_path, 'Train', 'JPEGImages',
                                                     im + "_crop" + str(crop_count) + '.png'), patch)
                        # Save crop mask
                        imageio.imwrite(os.path.join(output_path, 'Train', 'Masks',
                                                     im + "_crop" + str(crop_count) + '_mask.png'), patch_mask)
                    else:
                        # Save crop
                        imageio.imwrite(os.path.join(output_path, 'Validation', 'JPEGImages',
                                                     im + "_crop" + str(crop_count) + '.png'), patch)
                        # Save crop mask
                        imageio.imwrite(os.path.join(output_path, 'Validation', 'Masks',
                                                     im + "_crop" + str(crop_count) + '_mask.png'), patch_mask)
                    crop_count += 1


def create_dataset(input_path, train_image_list, validation_image_list, output_path, crop_size, stride_size):
    # Making sure output directory is created.
    check_mkdir(output_path)

    check_mkdir(os.path.join(output_path, 'Train'))
    check_mkdir(os.path.join(output_path, 'Validation'))

    folders = ['Train', 'Validation']
    subfolders = ['JPEGImages', 'Masks', 'Annotations', 'ImageSets', 'ReferencePoints']

    for f in folders:
        for sf in subfolders:
            check_mkdir(os.path.join(output_path, f, sf))
            if sf == 'ImageSets':
                if not os.path.exists(os.path.join(output_path, f, sf, 'Segmentation')):
                    os.mkdir(os.path.join(output_path, f, sf, 'Segmentation'))
                if not os.path.exists(os.path.join(output_path, f, sf, 'Main')):
                    os.mkdir(os.path.join(output_path, f, sf, 'Main'))

    process_images(input_path, train_image_list, output_path, crop_size, stride_size, is_train=True)
    process_images(input_path, validation_image_list, output_path, crop_size, stride_size, is_train=False)

