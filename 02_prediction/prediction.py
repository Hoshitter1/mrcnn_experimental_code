import os
import cv2
import tensorflow as tf
import time
from mrcnn import visualize
import mrcnn.model as modellib
from mrcnn.model import log
from mrcnn import utils
from mrcnn.config import Config
from mrcnn import visualize
import numpy as np
import matplotlib.pyplot as plt
import os

class TargetConfig(Config):
    # Give the configuration a recognizable name
    NAME = "target"
    GPU_COUNT = 1
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1
    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # Background + nails + hand
    # Number of training steps per epoch
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.3
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256


def output_image(img_list,mask_path,model_path):

    log_path = os.path.join(os.getcwd(),"logs")
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    #Read configurations
    config = TargetConfig()
    DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0
    start = time.time()
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=log_path,config=config)
    modellib_time = time.time()
    print("modellib_time:",modellib_time-start)

    start = time.time()
    #Read model
    model.load_weights(model_path, by_name=True)
    model_time = time.time()
    print("Time to read model:",model_time-start)
    class_names = ['BG', 'target', 'target_2']
    for img_path in img_list:
        print("*"*5 + "Info" + "*"*5 )
        print("file:",img_path)
        start = time.time()
        #Start to predict a nail
        image = cv2.imread(img_path)
        image_bb = image.copy()
        image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model.detect([image_RGB], verbose=0)
        r = results[0]
        predict_time = time.time()
        #Create Mask
        N = r['rois'].shape[0]
        print("N:",N)
        print("r[`class_ids`]:",r['class_ids'])
        print("r['rois'][i]",r['rois'])
        colors = visualize.random_colors(N)
        start = time.time()
        font = cv2.FONT_HERSHEY_SIMPLEX
        print("r['masks'].shape",r['masks'].shape)
        for i in range(N):
            mask = r['masks'][:, :, i]

            if r['class_ids'][i] == 0:
                print("class[0]_colors",colors[i])
                image = visualize.apply_mask(image, mask, [0,1,0]) #
            if r['class_ids'][i] == 1:
                print("class[1]_colors",colors[i])
                text = class_names[r['class_ids'][i]] + ':' + str(r['scores'][i])
                result_image = cv2.putText(image, text,
                                            (r['rois'][i][1],r['rois'][i][0]),
                                            font, 0.5,
                                            [0,255,0],
                                            1,
                                            cv2.LINE_AA
                                            )
                image = visualize.draw_box(image, r['rois'][i], [1,0,0])
                image = visualize.apply_mask(image, mask, [0,1,0]) #fingers="B"GR + image has to BGR not RGB
            if r['class_ids'][i] == 2:
                print("class[2]_colors:",colors[i])
                text = class_names[r['class_ids'][i]] + ':' + str(r['scores'][i])
                result_image = cv2.putText(image, text,
                                            (r['rois'][i][1],r['rois'][i][0]),
                                            font, 0.5,
                                            [0,0,255],
                                            1,
                                            cv2.LINE_AA
                                            )
                print("mask:",mask)
                image = visualize.apply_mask(image, mask, [0,0,1]) #nail=BG"R"
        fileName = os.path.join(mask_path,os.path.basename(img_path).split(".")[0] + "_mask.png")
        cv2.imwrite(fileName, image)
        drawing_time = time.time()
        print("drawing_time:",drawing_time-start)
        print("*"*10)
        # fileName_bb = os.path.join(mask_path,os.path.basename(img_path).split(".")[0] + "_mask_bb.png")
        # cv2.imwrite(fileName_bb, image_bb)
    return

def find_all_files(directory,extention="jpg"):
    filelist=[]
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file[-3:] == extention:
                filelist.append(os.path.join(root, file))
    return filelist


if __name__=="__main__":

    multi = True
    print("*"*10 + "start" + "*"*10)

    if multi == True:
        print("*"*10 + "mutil = True" + "*"*10)
        project_name = "03_test"
        c_dir = os.getcwd()
        base_dir = os.path.join(os.getcwd(),project_name)
        mask_path = os.path.join(base_dir,'result_mask_test_data')
        model_path = os.path.join(base_dir,'models','mask_rcnn_target_0010.h5')
        img_dir = os.path.join(base_dir,"img")
        img_list = find_all_files(img_dir,extention="jpg")
        output_image(img_list,mask_path,model_path)

    elif multi == "False":
        c_dir = os.getcwd()
        project_name = "something"
        img_name = "something"
        img_path = os.path.join(c_dir,project_name,"images",img_name)
        mask_path = os.path.join(os.getcwd(),'comp_0510','result_mask',"h100")
        model_path = os.path.join(os.getcwd(),'comp_0510','MRCNN_models','mask_rcnn_target_0100.h5')
        output_image(img_path,mask_path,model_path)
