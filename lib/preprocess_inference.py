import os
import cv2
import re
from tqdm import tqdm
import numpy as np

import os
import cv2
import re
from tqdm import tqdm

def process_data(lmks, image, target_size = 256):
    image_height, image_width, _ = image.shape
    anno_x = [lmks[i] for i in range(0, len(lmks)-1, 2)]
    anno_y = [lmks[i+1] for i in range(0, len(lmks)-1, 2)]
    bbox_xmin = min(anno_x)
    bbox_ymin = min(anno_y)
    bbox_xmax = max(anno_x)
    bbox_ymax = max(anno_y)
    bbox_width = bbox_xmax - bbox_xmin
    bbox_height = bbox_ymax - bbox_ymin
    scale = 1.1 
    bbox_xmin -= int((scale-1)/2*bbox_width)
    bbox_ymin -= int((scale-1)/2*bbox_height)
    bbox_width *= scale
    bbox_height *= scale
    bbox_width = int(bbox_width)
    bbox_height = int(bbox_height)
    bbox_xmin = max(bbox_xmin, 0)
    bbox_ymin = max(bbox_ymin, 0)
    bbox_width = min(bbox_width, image_width-bbox_xmin-1)
    bbox_height = min(bbox_height, image_height-bbox_ymin-1)
    # lmks = [[lmks[i], lmks[i+1]] for i in range(0,len(lmks)-1, 2)]
    bbox_xmin = int(bbox_xmin)
    bbox_ymin = int(bbox_ymin)
    bbox_width = int(bbox_width)
    bbox_height = int(bbox_height)
    anno = [[(lmks[i]-bbox_xmin)/bbox_width, (lmks[i+1]-bbox_ymin)/bbox_height] for i in range(0, len(lmks), 2)]
    bbox_xmax = bbox_xmin + bbox_width
    bbox_ymax = bbox_ymin + bbox_height
    image_crop = image[bbox_ymin:bbox_ymax, bbox_xmin:bbox_xmax, :]
    image_crop = cv2.resize(image_crop, (256,256))
    anno = anno[14:30]
    return image_crop, anno

def process_inference(lmks, image, target_size = 256):
    image_height, image_width, _ = image.shape
    lms = lmks[14:30]
    lms = [float(x) for x in lms]
    lms_x = lms[0::2]
    lms_y = lms[1::2]
    lms_x = [x if x >= 0 else 0 for x in lms_x]
    lms_x = [x if x <= image_width else image_width for x in lms_x]
    lms_y = [y if y >= 0 else 0 for y in lms_y]
    lms_y = [y if y <= image_height else image_height for y in lms_y]   
    lms = [[x,y] for x,y in zip(lms_x, lms_y)]
    lms = [x for z in lms for x in z] 
    min_x = min(lms_x)
    max_x = max(lms_x)
    min_y = min(lms_y)
    max_y = max(lms_y)
    width = max_x - min_x
    height = max_y - min_y
    center_x = (max_x + min_x) / 2
    center_y = (max_y + min_y) / 2

    if width > height:
        height = width
    else:
        width = height

    min_x = center_x - (width / 2)
    max_x = center_x + (width / 2)
    min_y = center_y - (height / 2)
    max_y = center_y + (height / 2)

    image_crop = image[int(min_y):int(max_y), int(min_x):int(max_x), :]
    
    image_crop = cv2.resize(image_crop, (256,256))

    tmp1 = [min_x, min_y]*8
    tmp1 = np.array(tmp1)
    tmp2 = [width, height]*8
    tmp2 = np.array(tmp2)
    lms = np.array(lms) - tmp1
    lms = lms/tmp2
    lms = lms.tolist()
    lms = zip(lms[0::2], lms[1::2])
    return image_crop, list(lms)
# Input method for lines:
with open("/home/kyv/Desktop/PIPNet/data/WFLW/WFLW_annotations/inference_annotation/test_inference.txt") as f:
    lines = f.readlines()
    
# Input method for images
mainfolder = "data/WFLW/WFLW_images/inference_images"
subfolders = [
    "kalpesh_15061980_171_session1",
    "paketa_24101994_180_session1",
    "simon_16101988_196_session2"
]
folderlist = []

for folder in subfolders:
    folder_path = os.path.join(mainfolder, folder)

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # Check if the current item is a file and ends with a recognized image extension
        if os.path.isfile(file_path) and any(file_path.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
            folderlist.append(file_path)

count = 0
with open("/home/kyv/Desktop/PIPNet/data/WFLW/test_2.txt", "w") as f:
    for i in tqdm(range(len(lines))):
        string = lines[i]
        # Extract file name and landmark coordinates
        string_list = string.strip().split()
        file_name = string_list[0]
        landmark_coordinates = [float(coord) for coord in string_list[1:]]
        # Find the corresponding image file
        image_path = None
        for file_path in folderlist:
            if file_name in file_path:
                image_path = file_path
                count+=1
                break

        if image_path is not None:
            image = cv2.imread(image_path)
            image_crop, anno = process_inference(landmark_coordinates, image)
            # Get the current working directory
            current_dir = os.getcwd()

            # Specify the relative path to the destination folder
            relative_path = "data/WFLW/images_test_3"

            # Combine the current directory and relative path
            destination_folder = os.path.join(current_dir, relative_path)

            # Update the image_crop_name with the new destination folder path
            image_crop_name = os.path.join(destination_folder, file_name + ".png")
            image_list = []
            cv2.imwrite(image_crop_name, image_crop)

            f.write(file_name + ' ')
            for coord in anno:
                f.write(str(coord[0]) + ' ' + str(coord[1]) + ' ')
            f.write("\n")

print(count)