import cv2
import os
import numpy as np
from pydub import AudioSegment
from pydub.playback import play

current_dir = os.getcwd()
annotation_path = "data/Inference/image_annotation/nofacebox.txt"
image_dir = "data/Inference/images"
destination_path = os.path.join(current_dir, annotation_path)
result_path = "data/Inference/image_annotation/boundingbox_result.txt"

# Create an empty list to store the bounding box coordinates
bounding_box_coordinates = []

def save_coordinates(event, x, y, flags, param):
    sound_1 = AudioSegment.from_file("/home/kyv/Desktop/NCAP_Visualization/data/beep-02.mp3")  # Replace "/path/to/sound.wav" with the actual path to your sound file
    sound = AudioSegment.from_file("/home/kyv/Desktop/PIPNet/data/Inference/Duck-quack.mp3")
    start_time = 0
    end_time = 1200
    sound_2 = sound[start_time:end_time]
    if event == cv2.EVENT_LBUTTONDOWN:
        bounding_box_coordinates.append(x)
        bounding_box_coordinates.append(y)
        play(sound_1)
        if len(bounding_box_coordinates) == 4:
            # Save the bounding box coordinates to the result file
            with open(result_path, "a") as file:
                filename = annotation.split()[0]
                coordinates = bounding_box_coordinates
                file.write(f"{filename} {' '.join(map(str, coordinates))}\n")
            bounding_box_coordinates.clear()
            play(sound_2)



# Read the annotations file
with open(destination_path, "r") as f:
    annotations = f.readlines()

# Iterate through the annotations
for annotation in annotations:
    # Get the filename and image path
    filename = annotation.split()[0]
    image_path = os.path.join(image_dir, filename)
    
    # Load the image
    img = cv2.imread(image_path)
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", save_coordinates)
    cv2.imshow("Image", img)

    # Wait until the bounding box coordinates are saved or "q" key is pressed
    while True:
        key = cv2.waitKey(1) & 0xFF

        # Check if the "q" key is pressed or the window is closed
        if key == ord("q") or cv2.getWindowProperty('Image', cv2.WND_PROP_VISIBLE) < 1:
            break

    # Close the window
    cv2.destroyAllWindows()

# Print a message indicating the process is complete
print("Bounding box coordinates saved for all images.")
