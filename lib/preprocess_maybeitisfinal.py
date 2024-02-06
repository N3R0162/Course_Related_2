import cv2
import os
import numpy as np
from tqdm import tqdm
def process_inference(lmks):
    lms = lmks[28:60]
    lms_float = [float(x) for x in lms]   
    lms_x = lms_float[0::2]
    lms_y = lms_float[1::2]
    lms = [[x,y] for x,y in zip(lms_x, lms_y)]
    lms = [x for z in lms for x in z]
    bbox = lmks[100:]
    bbox = [float(x) for x in bbox]
    bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = bbox
    
    width = bbox_xmax - bbox_xmin
    height = bbox_ymax - bbox_ymin
    scale = 1.2
    bbox_xmin -= width * (scale-1)/2
    bbox_ymin -= height * (scale-1)/2
    bbox_xmax += width * (scale-1)/2
    bbox_ymax += height * (scale-1)/2
    width = bbox_xmax - bbox_xmin
    height = bbox_ymax - bbox_ymin
    tmp1 = [bbox_xmin, bbox_ymin]*16
    tmp1 = np.array(tmp1)
    tmp2 = [width, height]*16
    tmp2 = np.array(tmp2)
    lms = np.array(lms) - tmp1
    lms = lms / tmp2
    lms = lms.tolist()
    lms = zip(lms[0::2], lms[1::2])
    #flatten the list
    lms = [x for z in lms for x in z]
    return lms

current_dir = os.getcwd()
relative_path = "data/Inference/image_annotation/annotation_result.txt"
with open(relative_path, "r") as f:
    for line in tqdm(f):
        line = line.strip()
        line = line.split()
        image_file = line[0]
        image_name = image_file.split('/')[-1]
        lmks = line[1:]
        landmarks = process_inference(lmks)
        #save the landmarks to the file
        with open("data/Inference/image_annotation/landmarks.txt", "a") as f:
            f.write(image_name + ' ' + ' '.join(map(str, landmarks)) + '\n')
    