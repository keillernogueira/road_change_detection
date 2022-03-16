#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python main.py --learning_rate 0.001 --epoch_num 20 40 80 --output_path output/scratch_1/ \
    --dataset_input_path /home/kno/road_detection/v2/ --dataset_create_path temp_dataset_192/ --crop_size 192 \
    --stride_size 150 --create_dataset True --operation Train

CUDA_VISIBLE_DEVICES=0 python main.py --learning_rate 0.001 --epoch_num 20 40 80 --output_path output/scratch_1/ \
    --dataset_input_path /home/kno/road_detection/v2/ --dataset_create_path temp_dataset_192/ --crop_size 192 \
    --stride_size 150 --create_dataset True --operation Train