import os
import cv2
import numpy as np

with open("/home/kyv/Desktop/PIPNet/data/WFLW/train.txt", "r") as f:
    annotations = f.readlines()

annotations = [x.strip().split() for x in annotations]

for label in annotations:

    image_name = label[0]
    lmks = label[1:]
    lmks = np.array([float(x) for x in lmks])
    landmarks = lmks.reshape(-1, 2)
    current_dir = os.getcwd()
    print("Image name: ",image_name)
    image_path = os.path.join(current_dir, "data/WFLW/images_train", image_name)
    img = cv2.imread(image_path)
    height, width, _ = img.shape

    for landmark in landmarks:
        x = int(landmark[0] * 256)
        y = int(landmark[1] * 256)
        print (x,y)
        cv2.circle(img, (x, y), 2, (0, 0, 255), 1)

    cv2.imshow("Facial Landmark", img)
    while True:
        key = cv2.waitKey(1)
        if key == 27 or cv2.getWindowProperty("Facial Landmark", cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
