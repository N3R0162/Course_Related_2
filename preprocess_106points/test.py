"""
Comparison size of two train txt
"""

import os
import cv2
DIR = os.getcwd()
image_folder = "/home/kyv/Desktop/PIPNet/data/WFLW/images_test"

# image_folder = "/home/kyv/Desktop/PIPNet/data/Inference/image_train"
for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #Turn RGB to Gray
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Check if image is RGB or grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        print(image.shape)
        print(f"{image_name} is RGB")
    elif len(image.shape) == 2:
        print(image.shape)
        print(f"{image_name} is grayscale")
    else:
        print(f"{image_name} has an unexpected shape: {image.shape}")


    cv2.imshow("image", image)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  
    cv2.destroyAllWindows()
