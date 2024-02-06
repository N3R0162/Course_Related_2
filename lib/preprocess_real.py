import os
import cv2
from tqdm import tqdm
current_directory = os.getcwd()
bbox_path = "data/Inference/image_annotation/annotation.txt"
bbox_file = os.path.join(current_directory, bbox_path)
result_name = "data/Inference/image_annotation/annotation_result.txt"
result_path = os.path.join(current_directory, result_name)
annotation_path = os.path.join(current_directory, "data/WFLW/WFLW_annotations/inference_annotation/test_inference.txt")

processed_image_path = os.path.join(current_directory, "data/Inference/image_train")
with open(bbox_file,"r") as f:
    bbox_list = f.readlines()
with open(annotation_path, "r") as f:
    annotation_list = f.readlines()

for annotation in tqdm(annotation_list):
    anno = annotation.strip().split(" ")
    anno_filename = anno[0]
    anno_value = anno[1:]
    for bbox in bbox_list:
        bbox_filename = bbox.strip().split(" ")[0] 
        bbox_value = bbox.strip().split(" ")[1:]
        if anno_filename == bbox_filename:
            anno_value.extend(bbox_value)
            anno_value = " ".join(anno_value)
            with open(result_path, "a") as f:
                f.write(anno_filename + " " + anno_value + "\n")
            break
            


