import cv2
import os

working_dir = os.getcwd()
image_path = os.path.join(working_dir, "data/WFLW/WFLW_images/inference_images/kalpesh_15061980_171_session1/kalpesh_15061980_171_session1_000076800.png")

img = cv2.imread(image_path)
cv2.imshow("Facial Landmark", img)
while True:
    key = cv2.waitKey(1)
    if key == 27 or cv2.getWindowProperty("Facial Landmark", cv2.WND_PROP_VISIBLE) < 1:
        break
    
cv2.destroyAllWindows()