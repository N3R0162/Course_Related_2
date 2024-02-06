import os
import cv2
import numpy as np


def get_landmarks(path):
    with open(path, "r") as f:
        landmarks = f.readlines()
        landmarks = [landmark.strip().split(' ')[1:-3] for landmark in landmarks]
        landmarks = [[float(x) for x in landmark] for landmark in landmarks]
        landmarks= np.array(landmarks, dtype=np.float32)
        return landmarks

# def preprocess_106

def gen_meanface(landmarks):
    meanface = np.mean(landmarks, axis=0)
    result = meanface.tolist()
    return result

def process_landmark(path):
    result = []
    with open(path, "r") as f:
        landmarks = f.readlines()
        landmarks_splitted = [landmark.strip().split(" ")[:-3] for landmark in landmarks]
        landmark_result = [" ".join(lmk for lmk in landmark) for landmark in landmarks_splitted]
    return landmark_result 

def main(root):
    landmarks = get_landmarks(root)
    processed_landmark = process_landmark(root)
    meanface = gen_meanface(landmarks)
    root_split = root.strip().split("/")
    
    print(f"MEANFACE TYPE: {type(meanface)}")

    meanface_path = "/".join(root_split[:-1])
    meanface_path = os.path.join(meanface_path, "meanface.txt")

    result_landmark_path = "/".join(root_split[:-1])
    result_landmark_path = os.path.join(result_landmark_path, "train.txt")

    with open(meanface_path, "w") as f:
        f.write(" ".join(str(x) for x in meanface))
        print("DONE")

    with open(result_landmark_path, "w") as f:
        f.write("\n".join(str(x) for x in processed_landmark))
        print("DONE")


if __name__ == "__main__":
    root = "/home/kyv/Desktop/PIPNet/data/106_points/train_original.txt"
    main(root)
    