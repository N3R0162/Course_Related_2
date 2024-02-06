import os
import cv2
import numpy as np

with open("data/WFLW/test_2.txt", "r") as f:
    annotations = f.readlines()

annotations = [x.strip().split() for x in annotations]
# mainfolder = "data/WFLW/WFLW_images/inference_images"
# subfolders = [
#     "simon_16101988_196_session2",
#     "paketa_24101994_180_session1",
#     "kalpesh_15061980_171_session1"
# ]
# folderlist = []
# for folder in subfolders:
#     folder_path = os.path.join(mainfolder, folder)

#     for filename in os.listdir(folder_path):
#         file_path = os.path.join(folder_path, filename)
#         # Check if the current item is a file and ends with a recognized image extension
#         if os.path.isfile(file_path) and any(file_path.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
#             folderlist.append(file_path)

folderlist = [filename for filename in os.listdir("data/Inference/images_test_3")]

for label in annotations:
    image_name = label[0]+".png"
    lmks = label[1:]
    lmks = np.array([float(x) for x in lmks])
    # print("*********************************")

    landmarks = lmks.reshape(-1, 2)

    image_path = None
    for file_path in folderlist:
        print(file_path)
        if image_name in file_path:
            # print(file_path)
            image_path = f"data/WFLW/images_test_3/{file_path}.png.png"
            break
    if image_path is not None:
        img = cv2.imread(image_path)
        height, width, _ = img.shape

        for landmark in landmarks:
            x = int(landmark[0])
            y = int(landmark[1])
            cv2.circle(img, (x, y), 2, (0, 0, 255), 1)

        cv2.imshow("Facial Landmark", img)
        while True:
            key = cv2.waitKey(1)
            if key == 27 or cv2.getWindowProperty("Facial Landmark", cv2.WND_PROP_VISIBLE) < 1:
                break

cv2.destroyAllWindows()
