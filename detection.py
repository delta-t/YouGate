import os
import cv2
import numpy as np
import time

from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from albumentations import pytorch


def get_model(num_classes):
    """Function to get built-in Mask-RCNN model"""
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model

def sharpen(image):
    """Filter to sharpen the image"""
    kernel = np.array([[-1, -1, -1], 
                        [-1, 9,-1], 
                        [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)

def find_corners(mask):
    """Function to draw rectangle on number plate
    and find its corners"""
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    rect = cv2.minAreaRect(cnts[0])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.fillPoly(mask, [box], (255, 255, 255))
    corners = cv2.goodFeaturesToTrack(mask, maxCorners=4, 
                                   qualityLevel=0.5, minDistance=10)
    return corners

def order_points(pts):
    """Function to order points:
    1. top-left
    2. top-right
    3. bottom-right
    4. bottom-left"""
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.squeeze().sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts.squeeze(), axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def warp_roi(image, corners, w, h):
    """Function to warp region of interests"""
    tl, tr, br, bl = corners
    dst = np.array([[0, 0], [w, 0],
                  [w, h], [0, h]], dtype = "float32")
    M = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl]), dst)
    warped = cv2.warpPerspective(image, M, (w, h))
    return warped

def predict(model, image_path, image_name):
    """Function for number plate detector prediction"""
    start_time = time.perf_counter()
    
    model.cuda()
    model.eval()

    image = cv2.imread(os.path.join(image_path, image_name))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0
    
    image_tensor = pytorch.ToTensor()(image=image)
    image_tensor = image_tensor["image"]
    image_tensor = image_tensor.cuda()
    
    outputs = model([image_tensor])
    outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]

    i = 0
    for (box, mask) in zip(outputs[0]["boxes"], outputs[0]["masks"]):
        box = box.detach().numpy()

        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
      
        h = y2 - y1
        w = x2 - x1

        mask = mask.detach().numpy().squeeze()
        mask *= 255
        mask = mask.astype(np.uint8)
        mask[np.where(mask >= 127)] = 255
        mask[np.where(mask < 127)] = 0

        corners = find_corners(mask)
        corners = order_points(corners)

        img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        img *= 255
        img = img.astype(np.uint8)

        cropped_img = cv2.bitwise_and(img, mask)
        warped_roi = warp_roi(cropped_img, corners, w, h)
        
        i += 1
        image_name = image_name.split(".")[0]
        cv2.imwrite(f"./detected_rois/{image_name}_{i}.jpg", warped_roi)
    end_time = time.perf_counter()

    return end_time - start_time