import cv2
import os
import numpy as np
meanface_path = os.path.join('data', 'Iris', 'meanface.txt')

img = np.zeros((256, 256, 3), dtype=np.uint8)

with open(meanface_path, 'r') as f:
    meanface = f.readlines()
meanface = [x.strip().split() for x in meanface]
meanface = [[float(x.rstrip(',')) for x in anno] for anno in meanface]
meanface = np.array(meanface)
meanface = meanface.reshape(-1, 2)
for landmark in meanface:
    x = int(landmark[0]*256)
    y = int(landmark[1]*256)
    cv2.circle(img, (x, y), 2, (0, 0, 255), 1)

cv2.imshow("Facial Landmark", img)
while True:
    key = cv2.waitKey(1)
    if key == 27 or cv2.getWindowProperty("Facial Landmark", cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows()