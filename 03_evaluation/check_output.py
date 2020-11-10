import os
import sys
import numpy as np
import tensorflow as tf

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
print(ROOT_DIR)
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import visualize
import mrcnn.model as modellib
from mrcnn.model import log
from mrcnn import utils
import cv2
import target
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
import time


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs",'test_december')
TARGET_WEIGHTS_PATH = os.path.join(ROOT_DIR, "logs",'dir1',"mask_rcnn_model_1.h5")
if not os.path.exists('./mask_outputs/'):
    os.mkdir('./mask_outputs/')
if not os.path.exists('./image_outputs/'):
    os.mkdir('./image_outputs/')


def check_outputs(weights_path=TARGET_WEIGHTS_PATH, visualizePerformanceForAll=False, datasetType="test",
                  evaluate=True):
    config = target.TargetConfig()
    TARGET_DIR = os.path.join(ROOT_DIR, "target/data/")

    # Override the training configurations with a few
    # changes for inferencing.
    class InferenceConfig(config.__class__):
        # Run detection on one image at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        DETECTION_MIN_CONFIDENCE = 0.6


    config = InferenceConfig()
    config.display()

    # Device to load the neural network on.
    # Useful if you're training a model on the same
    # machine, in which case use CPU and leave the
    # GPU for training.
    DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

    # Inspect the model in training or inference modes
    # values: 'inference' or 'training'
    TEST_MODE = "inference"

    # Load validation dataset
    dataset = target.TargetDataset()
    dataset.load_target(TARGET_DIR, datasetType)

    # Must call before using the dataset
    dataset.prepare()

    print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

    # Create model in inference mode
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                                  config=config)
    # Load weights
    print("Loading weights ", weights_path)
    start_model = time.time()
    model.load_weights(weights_path, by_name=True)
    elapsed_time_model = time.time() - start_model
    print ("elapsed_time_load_2:{0}".format(elapsed_time_model) + "[sec]")

    # save image w/ masks + masks
    if visualizePerformanceForAll:
        print("VISUALIZING DETECTED MASKS FOR ALL IMAGE IN TEST DATASET")
        for i in tqdm(range(len(dataset.image_ids))):
            image_id = i
            image, image_meta, gt_class_id, gt_bbox, gt_mask = \
                modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
            info = dataset.image_info[image_id]
            # Run object detection
            results = model.detect([image], verbose=0)
            r = results[0]

            h, w, c = r['masks'].shape
            canvas = np.zeros((h, w, 1))

            for i in range(c):
                mask = r['masks'][:, :, i].astype(np.uint8)
                #if r['class_ids'][i] == 1:
                    # finger
                    #canvas[:, :, i][np.where(mask == 1)] = 127
                if r['class_ids'][i] == 2:
                    canvas[:, :, 0][np.where(mask == 1)] = 255
            fileName = "./mask_outputs/" + info["id"]
            cv2.imwrite(fileName, canvas[:, :, ::-1].astype(np.uint8))

            ax = None

            masked_image = visualize.display_instances(image, r['rois'], (r['masks']), r['class_ids'],
                                                       dataset.class_names, r['scores'], ax=ax,
                                                       title="Predictions")#,show_mask=False, show_bbox=False)#,skipDisplay=True)#, skipBackground=True)
            fileName = "./image_outputs/" + info["id"]
            cv2.imwrite(fileName, masked_image[:,:,::-1].astype(np.uint8))

    if evaluate:
        print("------------------")
        print("RUNNING EVALUATION")
        print("------------------")
        # Load validation dataset
        if datasetType == 'test':
            pass
        else:
            dataset = target.TargetDataset()
            dataset.load_target(TARGET_DIR, "target")
            dataset.prepare()

        # Compute VOC-Style mAP @ IoU=0.5
        # Running on 10 images. Increase for better accuracy.
        imgNum = len(dataset.image_ids)
        #image_ids = np.random.choice(dataset.image_ids, imgNum)
        image_ids = [int(i) for i in range(imgNum)]
        APs = []
        precision = []
        recall = []
        load_1 = []
        load_2 = []
        for image_id in image_ids:
            print("image_id:",image_id)
            # Load image and ground truth data
            start = time.time()
            image, image_meta, gt_class_id, gt_bbox, gt_mask = \
                modellib.load_image_gt(dataset, InferenceConfig,
                                       image_id, use_mini_mask=False)
            elapsed_time_load = time.time() - start
            print ("elapsed_time_load:{0}".format(elapsed_time_load) + "[sec]")
            load_1.append(elapsed_time_load)
            start_2 = time.time()
            # Run object detection
            #print("image:",image)
            #print("[image]:"[image])
            results = model.detect([image], verbose=0)
            elapsed_time_load_2 = time.time() - start_2
            print ("elapsed_time_load_2:{0}".format(elapsed_time_load_2) + "[sec]")
            load_2.append(elapsed_time_load_2)
            r = results[0]
            # Compute AP
            AP, precisions, recalls, overlaps = \
                utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                 r["rois"], r["class_ids"], r["scores"], r['masks'])
            APs.append(AP)

        with open('result.csv', 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(precision)
            writer.writerows(recall)


if __name__ == '__main__':
    check_outputs(TARGET_WEIGHTS_PATH, datasetType="test",visualizePerformanceForAll=True, evaluate=False)
