"""
Mask R-CNN

------------------------------------------------------------

    # Train a new model starting from pre-trained COCO weights
    python3 target.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 target.py train --dataset=/path/to/balloon/dataset --weights=last

"""

import os, glob
import cv2_
import sys
import json
import datetime
import numpy as np
import skimage.draw
from PIL import Image
from imgaug import augmenters as iaa

# Root directory of the project
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model  as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")


# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
if not os.path.exists(DEFAULT_LOGS_DIR):
    os.mkdir(DEFAULT_LOGS_DIR)

############################################################
#  Configurations
############################################################


class TargetConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """

    # Give the configuration a recognizable name
    NAME = "target"
    GPU_COUNT = 1

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1
    # Number of classes (including background)
    NUM_CLASSES = 1 + 2# background

    # Number of training steps per epoch (number of images in val and train/bath number)
    STEPS_PER_EPOCH = int((220+40)/1)
    print(STEPS_PER_EPOCH)

    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

############################################################
#  Dataset
############################################################

class TargetDataset(utils.Dataset):

    def load_target(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("target", 1, "target")
        self.add_class("target_2", 2, "target_2")

        # Train or validation datasets
        assert subset in ["train", "val", "test", "accuracy_evaluation_warped", "accuracy_evaluation_cropped"]
        dataset_dir = os.path.join(dataset_dir, subset)
        images = os.listdir(dataset_dir)

        if subset == "train" or subset == "val" or subset == "test":
            # Add images
            path2json = os.path.join(dataset_dir, 'data.json')
            with open(path2json) as json_data:
                data = json.load(json_data)

            for a in data:
                self.add_image(
                    "target",
                    image_id=data[a]['fileName'],  # use file name as a unique image id
                    path=data[a]['path2img'],
                    path2mask=data[a]['path2mask'],
                    width=data[a]['width'],
                    height=data[a]['height']
                    )
        else:
            for fpath in glob.glob( "%s/*.jpg" % dataset_dir ):
                img = cv2.imread(fpath)
                height, width = img.shape[:2]
                new_height=int(height/4)
                new_width =int(width/4)
                img_cropped = img[new_height:new_height*2, new_width:new_width*2]
                self.add_image(
                    "target",
                    image_id=os.path.basename(fpath),  # use file name as a unique image id
                    path=fpath,
                    path2mask=fpath,# dummy
                    width=width,
                    height=height
                    )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        path2img = image_info['path']
        path2mask = image_info['path2mask']

        with Image.open(path2mask) as f:
            masks = np.array(f)
        masks = np.bitwise_not(masks.astype(np.bool))

        h, w, c = masks.shape
        if c == 3:
            target = masks[:, :, 1].reshape(h, w, 1)
            target2 = masks[:, :, 0].reshape(h, w, 1)
            mask = np.concatenate((target2,target), axis=2)
            # #green
            # if masks[0][0][0] == True and masks[0][0][1] == False and masks[0][0][2] ==True:
            #     target2 = masks[:, :, 0].reshape(h, w, 1)
            #     mask = np.concatenate((target2,target), axis=2)


        else:
            raise ValueError("The ndim of the mask is neither 3 or 4")

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        class_id = np.array([1, 2], dtype=np.int32)
        return mask, class_id

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        return info["path"]


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = TargetDataset()
    dataset_train.load_target(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = TargetDataset()
    dataset_val.load_target(args.dataset, "val")
    dataset_val.prepare()

    """
    # Set augmentation
    """
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)])
    ])
    # don't use
    # iaa.GaussianBlur(sigma=(0.0, 5.0)),
    # iaa.Multiply((0.8, 1.5))

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")

    #os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    #os.environ["CUDA_VISIBLE_DEVICES"]="0"
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=5,
                layers='heads')
    print("Train all layers")

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                # augmentation=augmentation,
                layers='all')

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse
    parent_dir = "/home/target_dir/"
    print("parent_dir",parent_dir)
    dataset = os.path.join(parent_dir,"00_create_data","dataset","train_val_test")
    print("dataset",dataset)
    # Parse command line argu/ments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect target.')
    parser.add_argument('--command',
                        metavar= "<command>",default="train",
                        help="'train' or 'target'")
    parser.add_argument('--dataset',
                        metavar="<dataset>", default=dataset,
                        help='Directory of the target dataset')
    parser.add_argument('--weights',
                        metavar="mask_rcnn_coco.h5",default="coco",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs',
                        default=DEFAULT_LOGS_DIR,
                        metavar="./logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = TargetConfig()
    else:
        class InferenceConfig(TargetConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        print("weights_path",weights_path)
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    #if args.weights.lower() == "coco":
    # Exclude the last layers because they require a matching
    # number of classes
    model.load_weights(weights_path, by_name=True, exclude=[
        "mrcnn_class_logits", "mrcnn_bbox_fc",
        "mrcnn_bbox", "mrcnn_mask"])
    #else:
    #    model.load_weights(weights_path, by_name=True)
    print('type(model): {}'.format(type(model)))

    # Train or evaluate
    if args.command == "train":
        train(model)
    else:
        print("'{}' is not recognized. "
              "Use 'train'".format(args.command))
