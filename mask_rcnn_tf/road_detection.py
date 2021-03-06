import os
import sys
import re
import numpy as np
import cv2
import gdal
import scipy

# Root directory of the project
# ROOT_DIR = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
# sys.path.append(ROOT_DIR)  # To find local version of the library
# sys.path.append("/home_hd/pedro/tf/erosion/mrcnn/")  # To find local version of the library

from mask_rcnn_tf.metrics import *

from mask_rcnn_tf.mrcnn.config import Config
from mask_rcnn_tf.mrcnn import utils
import mask_rcnn_tf.mrcnn.model as modellib


class RoadConfig(Config):
    # Give the configuration a recognizable name
    NAME = "road"

    # Train on 1 GPU and 5 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 5 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 5

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 road

    IMAGE_MIN_DIM = 448
    IMAGE_MAX_DIM = 448

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 448)  # anchor side in pixels

    # USE_MINI_MASK = False

    TRAIN_ROIS_PER_IMAGE = 32
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 5


class InferenceConfig(RoadConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class RoadDataset(utils.Dataset):
    """
    Override function to load masks, and image reference (Obrigatory)
    and create funcion to load the dataset
    """

    def load_dataset(self, source_list, size):
        # Add Classes
        self.add_class("road-images", 1, "road")

        # Register Images
        ids = 0
        for sid, source in enumerate(source_list):
            part = os.path.split(os.path.split(source)[0])[-1]
            image_dir = os.path.join(source, "JPEGImages")
            mask_dir = os.path.join(source, "Masks")
            files = os.listdir(image_dir)
            for f in files:
                print(f)
                if '.png' in f:
                    name = f.split('.')[0]
                    mask = name + "_mask.png"
                    img_source = f.split('_')[0]
                    refid = f.split('_')[1]

                    self.add_image("road-images", ids, os.path.join(image_dir, f), width=size, height=size,
                                   part=part, name=name, mask=mask, refid=refid, refimg=img_source,
                                   img_source=image_dir, mask_source=mask_dir, sourcedir=source, sid=sid)
                    ids += 1

    def load_mask(self, image_id):
        """
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        mask_source = info['mask_source']
        mask_file = info['mask']
        mask = (cv2.imread(os.path.join(mask_source, mask_file), 0) == 1).astype(bool)

        return np.stack([mask], axis=2), np.asarray([1])

    def image_reference(self, image_id):
        return "road-images::{}".format(image_id)

    def get_image_info(self, image_id):
        return self.image_info[image_id]


def train(input_path, output_path, model_weights, learning_rate, epochs, crop_size):
    INITIAL_WEIGHTS = model_weights  # os.path.join(os.path.dirname(__file__), 'mask_rcnn_balloon.h5')

    config = RoadConfig()
    config.IMAGE_MAX_DIM = crop_size
    config.IMAGE_MIN_DIM = crop_size
    config.RPN_ANCHOR_SCALES = (crop_size/14.0, crop_size/7.0, crop_size/3.5, crop_size/1.75, crop_size)
    config.display()

    # Training dataset
    dataset_train = RoadDataset()
    dataset_train.load_dataset(os.path.join(input_path, 'Train'), crop_size)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = RoadDataset()
    dataset_val.load_dataset(os.path.join(input_path, 'Validation'), crop_size)
    dataset_val.prepare()

    model = modellib.MaskRCNN(mode="training", config=config, model_dir=output_path)
    model.load_weights(INITIAL_WEIGHTS, by_name=True)

    # Training - Stage 1
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=learning_rate,
                epochs=epochs[0],
                layers='heads')

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=learning_rate,
                epochs=epochs[1],
                layers='4+')

    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=learning_rate / 10,
                epochs=epochs[2],
                layers='all')

    # elif process == 'fnt':
    #     TRAINED_MODEL = getLastWeights(prev_model_weights)
    #     config = ErosionConfig()
    #     config.display()
    #
    #     # Training dataset
    #     dataset_train = ErosionDataset()
    #     dataset_train.load_dataset(train_folders, IMG_SIZE)
    #     dataset_train.prepare()
    #
    #     # Validation dataset
    #     dataset_val = ErosionDataset()
    #     dataset_val.load_dataset(val_folders, IMG_SIZE)
    #     dataset_val.prepare()
    #
    #     model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
    #     model.load_weights(TRAINED_MODEL, by_name=True)
    #
    #     # Fine tune tune network heads
    #     print("Fine tune network heads")
    #     model.train(dataset_train, dataset_val,
    #                 learning_rate=lr / 10,
    #                 epochs=epochs[0],
    #                 layers='heads')
    #
    # elif process == 'validating':
    #     VISUAL_DIR = os.path.join(outputfolder, "visualization")
    #     if not os.path.exists(VISUAL_DIR):
    #         os.mkdir(VISUAL_DIR)
    #
    #     # TODO Metricas
    #     inference_config = InferenceConfig()
    #
    #     # Testing dataset
    #     dataset_val = ErosionDataset()
    #     dataset_val.load_dataset(val_folders, IMG_SIZE)
    #     dataset_val.prepare()
    #     if complete_vmap:
    #         referecepoints = getReferencePoints(val_folders)
    #         referenceimgspaths = getReferenceImagesPaths(val_folders)
    #         completepreds = getEmptyPredctionMaps(referenceimgspaths)
    #
    #     model_path = getLastWeights(prev_model_weights)
    #
    #     # Recreate the model in inference mode
    #     model = modellib.MaskRCNN(mode="inference",
    #                               config=inference_config,
    #                               model_dir=MODEL_DIR)
    #
    #     # Load trained weights
    #     print("Loading weights from ", model_path)
    #     model.load_weights(model_path, by_name=True)
    #
    #     # Utilitary arrays
    #     APs = []
    #     gt_boxes = []
    #     pred_boxes = []
    #     gts = []
    #     preds = []
    #     scores = []
    #
    #     for image_id in dataset_val.image_ids:
    #         # Load image and ground truth data
    #         image, image_meta, gt_class_id, gt_box, gt_mask =\
    #         modellib.load_image_gt(dataset_val, inference_config,
    #                             image_id, use_mini_mask=False)
    #
    #         image_info = dataset_val.get_image_info(image_id)
    #
    #         molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    #
    #         print(image_info['name'])
    #
    #         # Run object detection
    #         results = model.detect([image], verbose=0)
    #         r = results[0]
    #
    #         # O algoritmo filtra mascaras completamente zeradas,
    #         # por isso os elses abaixo para gerar ground truths nesses casos
    #         AP = 0.0
    #         computeAp = True
    #         if gt_box.shape[0] == 0:
    #             computeAp = False
    #             gt_box = np.asarray([[0, 0, 0, 0]])
    #         if gt_mask.shape[-1] == 0:
    #             gt_mask = np.stack([np.zeros((gt_mask.shape[0],gt_mask.shape[1]))], axis=2)
    #
    #         if r['rois'].shape[0] == 0:
    #             if not computeAp:
    #                 AP = 1.0
    #             r['rois'] = np.asarray([[0, 0, 0, 0]])
    #
    #         if r['masks'].shape[-1] == 0:
    #             r['masks'] = np.stack([np.zeros((gt_mask.shape[0],gt_mask.shape[1]))], axis=2)
    #
    #         if r['scores'].shape[0] == 0:
    #             r['scores'] = np.asarray([0.0])
    #
    #         gt_bbox = gt_box[0]
    #         gt = gt_mask[:,:,0]
    #
    #         pred_bbox = r['rois'][0]
    #         pred = r['masks'][:,:,0]
    #         scr = r['scores'][0]
    #
    #         gt_boxes.append(gt_bbox)
    #         pred_boxes.append(pred_bbox)
    #         gts.append(gt)
    #         preds.append(pred)
    #         scores.append(scr)
    #
    #         _cm_det = metrics.compute_confusion_matrix_detection(np.asarray([gt_bbox]), [pred_bbox], [scr])
    #         _cm_seg = metrics.compute_confusion_matrix_segmentation(np.asarray([gt]), np.asarray([pred]), [scr])
    #
    #         _visual_pred = metrics.create_simple_visualization(pred, pred_bbox)
    #         visual_pred_file = os.path.join(VISUAL_DIR, image_info['name'] + 'visual_pred.png')
    #
    #         _visual_map = metrics.create_map_visualization(gt, pred, scr)
    #         visual_map_file = os.path.join(VISUAL_DIR, image_info['name'] + 'visual_map.png')
    #
    #         _visual_overlay = metrics.create_overlay_visualization(image, pred, pred_bbox, scr)
    #         visual_overlay_file = os.path.join(VISUAL_DIR, image_info['name'] + 'visual_overlay.png')
    #
    #         saveImg(_visual_map, visual_map_file)
    #         saveImg(_visual_overlay, visual_overlay_file)
    #         saveImg(_visual_pred, visual_pred_file)
    #
    #         if complete_vmap:
    #             sid = image_info['sid']
    #             rid = image_info['refid']
    #             rfid = image_info['refimg']
    #             metrics.update_complete_visual_map(completepreds[rfid], pred, scr, referecepoints, rfid, rid)
    #
    #         # Compute AP
    #         if computeAp:
    #             AP, precisions, recalls, overlaps = utils.compute_ap(gt_box, gt_class_id, gt_mask,
    #                                             r["rois"], r["class_ids"], r["scores"], r['masks'])
    #
    #         APs.append(AP)
    #
    #         print("Metrics for image[{0}] : {1}".format(image_id, image_info['name']))
    #         print("----------- Detection ----------- ")
    #         print("Confusion Matrix= " + np.array_str(_cm_det).replace("\n", ""))
    #         print("AP= " + str(AP))
    #         metrics.evaluate(_cm_seg, pred=pred, gt=gt, Message="Segmentation")
    #
    #
    #     cm_det = metrics.compute_confusion_matrix_detection(np.asarray(gt_boxes), np.asarray(pred_boxes), scores)
    #     cm_seg = metrics.compute_confusion_matrix_segmentation(np.asarray(gts), np.asarray(preds), scores)
    #     metrics.evaluate(cm_seg)
    #     print("----------- Detection ----------- ")
    #     print("Confusion Matrix= " + np.array_str(cm_det).replace("\n", ""))
    #     print("mAP= ", np.mean(APs))
    #
    #     if complete_vmap:
    #         for iname, pred in completepreds.items():
    #             fname = os.path.join(VISUAL_DIR, iname + "_pred.png")
    #             fnamecolor = os.path.join(VISUAL_DIR, iname + "_pred_color.png")
    #             vmap = metrics.save_complete_visual_map(pred, fname)
    #
    # elif process == 'testing':
    #     VISUAL_DIR = os.path.join(outputfolder)
    #     if not os.path.exists(VISUAL_DIR):
    #         os.mkdir(VISUAL_DIR)
    #
    #     # TODO Metricas
    #     inference_config = InferenceConfig()
    #
    #     # Testing dataset
    #     dataset_test = ErosionDataset()
    #     dataset_test.load_dataset(test_folders, IMG_SIZE)
    #     dataset_test.prepare()
    #     if complete_vmap:
    #         referecepoints = getReferencePoints(test_folders)
    #         referenceimgspaths = getReferenceImagesPaths(test_folders)
    #         completepreds = getEmptyPredctionMaps(referenceimgspaths)
    #     print ("_________________________________________________________________________________________________")
    #     print (referenceimgspaths)
    #     model_path = getLastWeights(prev_model_weights)
    #
    #     # Recreate the model in inference mode
    #     model = modellib.MaskRCNN(mode="inference",
    #                               config=inference_config,
    #                               model_dir=MODEL_DIR)
    #
    #     # Load trained weights
    #     print("Loading weights from ", model_path)
    #     model.load_weights(model_path, by_name=True)
    #
    #
    #     # Utilitary arrays
    #     APs = []
    #     gt_boxes = []
    #     pred_boxes = []
    #     gts = []
    #     preds = []
    #     scores = []
    #
    #     for image_id in dataset_test.image_ids:
    #         # Load image and ground truth data
    #         image, image_meta, gt_class_id, gt_box, gt_mask =\
    #         modellib.load_image_gt(dataset_test, inference_config,
    #                             image_id, use_mini_mask=False)
    #
    #         image_info = dataset_test.get_image_info(image_id)
    #
    #         molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    #
    #         # Run object detection
    #         results = model.detect([image], verbose=0)
    #         r = results[0]
    #
    #         computeAp = True
    #         if gt_box.shape[0] == 0:
    #             computeAp = False
    #             gt_box = np.asarray([[0, 0, 0, 0]])
    #         if gt_mask.shape[-1] == 0:
    #             gt_mask = np.stack([np.zeros((gt_mask.shape[0],gt_mask.shape[1]))], axis=2)
    #
    #         if r['rois'].shape[0] == 0:
    #             if not computeAp:
    #                 AP = 1.0
    #             r['rois'] = np.asarray([[0, 0, 0, 0]])
    #
    #         if r['masks'].shape[-1] == 0:
    #             r['masks'] = np.stack([np.zeros((gt_mask.shape[0],gt_mask.shape[1]))], axis=2)
    #
    #         if r['scores'].shape[0] == 0:
    #             r['scores'] = np.asarray([0.0])
    #
    #         gt_bbox = gt_box[0]
    #         gt = gt_mask[:,:,0]
    #
    #         pred_bbox = r['rois'][0]
    #         pred = r['masks'][:,:,0]
    #         scr = r['scores'][0]
    #
    #         gt_boxes.append(gt_bbox)
    #         pred_boxes.append(pred_bbox)
    #         gts.append(gt)
    #         preds.append(pred)
    #         scores.append(scr)
    #
    #         sid = image_info['sid']
    #         rid = image_info['refid']
    #         rfid = image_info['refimg']
    #         metrics.update_complete_visual_map(completepreds[rfid], pred, scr, referecepoints, rfid, rid)
    #
    #
    #     shapenames = []
    #     for iname, pred in completepreds.items():
    #         fname = os.path.join(VISUAL_DIR, iname + "_pred.tif")
    #         fnamecolor = os.path.join(VISUAL_DIR, iname + "_pred_color.png")
    #         vmap = metrics.save_complete_visual_map_as_tif(pred, fname, referenceimgspaths[iname])
    #         shpname = metrics.create_shapefile(VISUAL_DIR, fname)
    #         os.remove(fname)
    #         shapenames.append(shpname)
    #
    #         #cvmap = metrics.create_map_visualization(completegts[x], vmap, 1.0)
    #         #saveImgLarge(cvmap, fnamecolor)
    #
    #     if outshpname is not '':
    #         if '.shp' not in outshpname:
    #             outshpname += '.shp'
    #         merged = metrics.merge_shapefiles(shapenames, VISUAL_DIR, outshpname)
    #         print(merged)
