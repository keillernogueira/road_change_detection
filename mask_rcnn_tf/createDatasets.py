import os
import imageio
import tifffile
import numpy as np

from utils import check_mkdir


def compute_image_mean(data):
    _mean = np.mean(np.mean(np.mean(data, axis=0), axis=0), axis=0)
    _std = np.std(data, axis=0, ddof=1)[0, 0, :]

    return _mean, _std


def process_images(input_path, image_list, output_path, crop_size, stride_size, is_train, calculate_mean=False):
    all_patches = []

    # Loop trough images and create crops
    for im in image_list:
        print("Processing image: " + im)

        # Open image
        img = tifffile.imread(os.path.join(input_path, 'area' + im + '_landsat8_toa_2013_pansharpen.tif'))
        img[np.where(np.isnan(img))] = 0  # replace nan with zero
        mask = (imageio.imread(os.path.join(input_path, 'area' + im + '_mask.png'))).astype(np.uint8)
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
                    patch = img[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :]
                elif len(patch) != crop_size:
                    cur_x = cur_x - (crop_size - len(patch))
                    patch = img[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :]
                elif len(patch[0]) != crop_size:
                    cur_y = cur_y - (crop_size - len(patch[0]))
                    patch = img[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :]

                assert patch.shape[0] == crop_size, "1 Error create_distributions_over_classes: " \
                                                    "Current patch size is " + str(len(patch)) + \
                                                    "x" + str(len(patch[0]))
                assert patch.shape[1] == crop_size, "2 Error create_distributions_over_classes: " \
                                                    "Current patch size is " + str(len(patch)) + \
                                                    "x" + str(len(patch[0]))

                if calculate_mean:
                    all_patches.append(patch)

                patch_mask = mask[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]

                if is_train:
                    # we need to check that the patch has road in it
                    count = np.bincount(patch_mask.astype(int).flatten())
                    # print(type(patch), patch.shape, type(patch_mask), patch_mask.shape, count)
                    if len(count) == 2:
                        # Save crop
                        # options to save cv2.imwrite, imageio.imwrite, tifffile.imwrite
                        tifffile.imwrite(os.path.join(output_path, 'Train', 'JPEGImages',
                                                      im + "_crop" + str(crop_count) + '.png'), patch)
                        # Save crop mask
                        imageio.imwrite(os.path.join(output_path, 'Train', 'Masks',
                                                     im + "_crop" + str(crop_count) + '_mask.png'), patch_mask)
                        crop_count += 1
                else:
                    # Save crop
                    # options to save cv2.imwrite, imageio.imwrite, tifffile.imwrite
                    tifffile.imwrite(os.path.join(output_path, 'Validation', 'JPEGImages',
                                                  im + "_crop" + str(crop_count) + '.png'), patch)
                    # Save crop mask
                    imageio.imwrite(os.path.join(output_path, 'Validation', 'Masks',
                                                 im + "_crop" + str(crop_count) + '_mask.png'), patch_mask)
                    crop_count += 1

    if calculate_mean:
        _mean, _std = compute_image_mean(all_patches)
        print(_mean, _std)
        np.save(os.path.join(output_path, 'crop_' + str(crop_size) + '_stride_' +
                             str(stride_size) + '_mean.npy'), _mean)
        np.save(os.path.join(output_path, 'crop_' + str(crop_size) + '_stride_' +
                             str(stride_size) + '_std.npy'), _std)


def create_dataset(input_path, train_image_list, validation_image_list, output_path, crop_size, stride_size,
                   calculate_mean=False):
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

    process_images(input_path, train_image_list, output_path, crop_size, stride_size, is_train=True,
                   calculate_mean=calculate_mean)
    process_images(input_path, validation_image_list, output_path, crop_size, stride_size, is_train=False)

