import streamlit as st 
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import av
import threading

import cv2, os
import numpy as np
import importlib
from math import floor

import sys
sys.path.insert(0, 'FaceBoxesV2')
sys.path.insert(0, '..')
from faceboxes_detector import *

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from networks import *
import data_utils
from functions import *

# os.environ['TWILIO_ACCOUNT_SID'] = 'AC38b506e97c51e0bc3de0a030f12cf2d4'
# os.environ['TWILIO_AUTH_TOKEN'] = '1c58c61779cb60f16861f5e3d090abcb'    

#Init model variables:
experiment_name = "pip_32_16_60_r18_l2_l1_10_1_nb10"
data_name = "WFLW"
config_path = '.experiments.{}.{}'.format(data_name, experiment_name)
video_file = "/dev/video2"  #Camera_path
save_dir = "snapshots/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10"  #Weight_path

my_config = importlib.import_module(config_path, package='PIPNet')
Config = getattr(my_config, 'Config')
cfg = Config()
cfg.experiment_name = experiment_name
cfg.data_name = data_name
meanface_indices, reverse_index1, reverse_index2, max_len = get_meanface(os.path.join('data', cfg.data_name, 'meanface.txt'), cfg.num_nb)

# Define a unique cache key for session-specific caching
cache_key = "pipnet_model"

# Check if the model has already been loaded and stored in the session state
if cache_key in st.session_state:
    net = st.session_state[cache_key]  # Retrieve the model from session state
else:
    # Load the model and its weights
    resnet18 = models.resnet18(pretrained=cfg.pretrained)
    net = Pip_resnet18(resnet18, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
    # Set device (CPU/GPU)
    if cfg.use_gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    net = net.to(device)
    
    # Load the state dictionary
    weight_file = os.path.join(save_dir, 'epoch%d.pth' % (cfg.num_epochs-1))
    state_dict = torch.load(weight_file, map_location=device)
    net.load_state_dict(state_dict)
    
    # Store the loaded model in the session state for future use
    st.session_state[cache_key] = net

logger = logging.getLogger(__name__)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([transforms.Resize((cfg.input_size, cfg.input_size)), transforms.ToTensor(), normalize])


#Page config (icon, title)
st.set_page_config(page_title="Streamlit WebRTC Demo", page_icon=":shark:", layout="centered", initial_sidebar_state="auto")        

#Sidebar:
task_list = ["Video Stream", "Pictures"]
with st.sidebar:
    st.title("PIPNet")
    task_name = st.selectbox("Choose a task", task_list)
st.title(task_name)    
eye_aspect_ratio = None

def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def play_webcam(model):
    webcam_path = "/dev/video2"







# Define a callback function to process each frame
def callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")

    # Run your model and draw bounding boxes here
    # Modify this part to fit your model and drawing logic
    # For example:
    # detections = model.detect_objects(img)
    # for detection in detections:
    #     x, y, w, h = detection["bbox"]
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    frame_height, frame_width = img.shape[:2]
    sleepy_frames = 0
    detector = FaceBoxesDetector('FaceBoxes', '/home/kyv/WD_500G/Project/PIPNet/FaceBoxesV2/weights/FaceBoxesV2.pth', True, device=torch.device("cuda:0"))
    my_thresh = 0.9
    det_box_scale = 1.2
    net.eval()    
    detections, _ = detector.detect(img, my_thresh, 1)
    for i in range (len(detections)):
        det_xmin = detections[i][2]
        det_ymin = detections[i][3]
        det_width = detections[i][4]
        det_height = detections[i][5]
        det_xmax = det_xmin + det_width - 1
        det_ymax = det_ymin + det_height - 1

        det_xmin -= int(det_width * (det_box_scale-1)/2)
        # remove a part of top area for alignment, see paper for details
        det_ymin += int(det_height * (det_box_scale-1)/2)
        det_xmax += int(det_width * (det_box_scale-1)/2)
        det_ymax += int(det_height * (det_box_scale-1)/2)
        det_xmin = max(det_xmin, 0)
        det_ymin = max(det_ymin, 0)
        det_xmax = min(det_xmax, frame_width-1)
        det_ymax = min(det_ymax, frame_height-1)
        det_width = det_xmax - det_xmin + 1
        det_height = det_ymax - det_ymin + 1
        cv2.rectangle(img, (det_xmin, det_ymin), (det_xmax, det_ymax), (0, 0, 255), 2)
        det_crop = img[det_ymin:det_ymax, det_xmin:det_xmax, :]
        det_crop = cv2.resize(det_crop, (256, 256))
        inputs = Image.fromarray(det_crop[:,:,::-1].astype('uint8'), 'RGB')
        inputs = preprocess(inputs).unsqueeze(0)
        inputs = inputs.to(torch.device("cuda:0"))
        lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = forward_pip(net, inputs, preprocess, 256, 32, 10)
        lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=1).flatten()
        tmp_nb_x = lms_pred_nb_x[reverse_index1, reverse_index2].view(16, max_len)
        tmp_nb_y = lms_pred_nb_y[reverse_index1, reverse_index2].view(16, max_len)
        tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1,1)
        tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1,1)
        lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten()
        lms_pred = lms_pred.cpu().numpy()
        lms_pred_merge = lms_pred_merge.cpu().numpy()
        # Extract relevant features and classify user as sleepy or not
        aspect_ratio = calculate_aspect_ratio(lms_pred_merge)
        for i in range(16):
            x_pred = lms_pred_merge[i*2] * det_width
            y_pred = lms_pred_merge[i*2+1] * det_height
            cv2.circle(img, (int(x_pred)+det_xmin, int(y_pred)+det_ymin), 1, (0, 0, 255), -1)


    # Return the processed frame
    return av.VideoFrame.from_ndarray(img, format="bgr24")



# Set up the Streamlit WebRTC connection
if __name__ == "__main__":
    # ... setup code ...
    # Set up the Streamlit WebRTC connection
    webrtc_ctx = webrtc_streamer(
        key="PIPNet",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
