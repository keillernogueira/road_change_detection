import argparse
import sys
import os

from utils import check_mkdir
from mask_rcnn_tf.createDatasets import create_dataset
from mask_rcnn_tf.road_detection import train


def main():
    parser = argparse.ArgumentParser(description='main')
    # general options
    parser.add_argument('--operation', type=str, required=True,
                        help='Operation [Options: Train | Test]')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save outcomes (such as images and trained models) of the algorithm.')
    parser.add_argument('--simulate_dataset', type=str2bool, default=False,
                        help='Used to speed up the development process.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU number.')
    parser.add_argument('--save_images', type=str2bool, default=False, help='Bool to save images.')

    # dataset options
    parser.add_argument('--dataset', type=str, help='Dataset [Options: road_detection].')
    parser.add_argument('--dataset_input_path', type=str, help='Dataset path.')
    # parser.add_argument('--dataset_gt_path', type=str, help='Ground truth path.')
    parser.add_argument('--dataset_create_path', type=str, required=True,
                        help='Path to save the created dataset.')
    parser.add_argument('--num_classes', type=int, help='Number of classes.')

    # model options
    parser.add_argument('--model_name', type=str, default='dilated_grsl_rate8',
                        help='Model to test [Options: dilated_grsl_rate8]')
    parser.add_argument('--model_path', type=str, default=None, help='Model path.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    # parser.add_argument('--weight_decay', type=float, default=0.005, help='Weight decay')
    # parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epoch_num', type=int, nargs='+', required=True, help='Number of epochs')
    # parser.add_argument('--loss_weight', type=float, nargs='+', default=[1.0, 1.0], help='Weight Loss.')
    parser.add_argument('--crop_size', type=int, default=25, help='Reference crop size.')
    parser.add_argument('--stride_crop', type=int, default=15, help='Reference crop stride')

    args = parser.parse_args()
    # Making sure output directory is created.
    check_mkdir(args.output_path)
    print(args)

    # Setting datasets.
    create_dataset(args.dataset_input_path, ['2', '3'], ['4'], args.dataset_create_path,
                   args.crop_size, args.stride_size)

    train(args.dataset_create_path, args.output_path, args.model_path,
          args.learning_rate, args.epoch_num, args.crop_size)


if __name__ == "__main__":
    main()
