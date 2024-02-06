import numpy as np
import cv2
import os

path ="./data/LaPa/train/landmarks"
file_name = os.listdir(path)

for filename in file_name:
    with open(os.path.join(path, filename), "r") as f:
        coordinates = f.readlines()

    # Create a blank image
    width, height = 500, 500
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Loop through the coordinates and draw circles at each point
    for coord in coordinates[1:]:
        coordlist = coord.split(" ")
        x = float(coordlist[0])
        y = float(coordlist[1].strip())  # Remove newline character
        cv2.circle(image, (int(x), int(y)), 3, (0, 0, 255), -1)

    # Display the image
    cv2.imshow("Coordinates Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
