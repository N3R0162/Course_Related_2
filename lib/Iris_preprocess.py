import os, cv2
import hdf5storage
import numpy as np
import sys
from tqdm import tqdm

DIR = os.getcwd()
def processIris(root, data_folder, img_name, label_name, target_size):
    img_folder = os.path.join(root, data_folder, 'images')
    label_folder = os.path.join(root, data_folder, 'labels')
    out_folder = os.path.join(root, data_folder, 'out')
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    with open(os.path.join(label_folder, label_name)) as f:
        annos = f.readlines()

    annos_new = []
    for anno in annos:
        anno = anno.strip().split(' ')
        img_path = os.path.join(img_folder, anno[0])
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        landmarks = []
        # extract eye landmarks
        landmarks.append(anno[36*2])
        landmarks.append(anno[36*2+1])
        landmarks.append(anno[37*2])
        landmarks.append(anno[37*2+1])
        landmarks.append(anno[38*2])
        landmarks.append(anno[38*2+1])
        landmarks.append(anno[39*2])
        landmarks.append(anno[39*2+1])
        landmarks.append(anno[42*2])
        landmarks.append(anno[42*2+1])
        landmarks.append(anno[43*2])
        landmarks.append(anno[43*2+1])
        landmarks.append(anno[44*2])
        landmarks.append(anno[44*2+1])
        landmarks.append(anno[45*2])
        landmarks.append(anno[45*2+1])
        landmarks.append(anno[46*2])
        landmarks.append(anno[46*2+1])
        landmarks.append(anno[47*2])
        landmarks.append(anno[47*2+1])
        landmarks.append(anno[48*2])
        landmarks.append(anno[48*2+1])
        landmarks.append(anno[49*2])
        landmarks.append(anno[49*2+1])
        landmarks.append(anno[50*2])
        landmarks.append(anno[50*2+1])
        # write landmarks to file
        out_path = os.path.join(out_folder, anno[0].replace('.jpg', '.txt'))
        with open(out_path, 'w') as f:
            for i in range(len(landmarks)):
                f.write(str(float(landmarks[i]) / w) + ' ' + str(float(landmarks[i+1]) / h) + '\n')
                i += 1
        annos_new.append(landmarks)

    return annos_new