import cv2, os
import sys
sys.path.insert(0, 'FaceBoxesV2')
sys.path.insert(0, '..')
import numpy as np
import pickle
import importlib
from math import floor
from faceboxes_detector import *
import time

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

import streamlit as st
import PIL

#Init variables:
experiment_name = "pip_32_16_60_r18_l2_l1_10_1_nb10"
data_name = "WFLW"
config_path = '.experiments.{}.{}'.format(data_name, experiment_name)
video_file = "/dev/video2"

#Init config (path: "experiments/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10.py")
my_config = importlib.import_module(config_path, package='PIPNet')
Config = getattr(my_config, 'Config')
cfg = Config()
cfg.experiment_name = experiment_name
cfg.data_name = data_name

#Init weight path:
save_dir = "snapshots/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10"
weight_file = os.path.join(save_dir, 'epoch%d.pth' % (cfg.num_epochs-1))

#Get meanface:
meanface_indices, reverse_index1, reverse_index2, max_len = get_meanface(os.path.join('data', cfg.data_name, 'meanface.txt'), cfg.num_nb)

#Init model:
resnet18 = models.resnet18(pretrained=cfg.pretrained)
net = Pip_resnet18(resnet18, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
if cfg.use_gpu:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
net = net.to(device)
weight_file = os.path.join(save_dir, 'epoch%d.pth' % (cfg.num_epochs-1))
state_dict = torch.load(weight_file, map_location=device)
net.load_state_dict(state_dict)

#Preprocess frame:
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([transforms.Resize((cfg.input_size, cfg.input_size)), transforms.ToTensor(), normalize])


def demo_video_sleepy(video_file, net, preprocess, input_size, net_stride, num_nb, use_gpu, device):
    

#====TODO: Create UI=====
st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon=":sleeping:",
    layout="wide",
    initial_sidebar_state="expanded",
)

#Sidebar:
with st.sidebar:
    st.header("Image/Video Config")
    source_image = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    source_video = st.sidebar.file_uploader("Upload Video", type=["mp4"])
    st.header("Model Config")
    model = st.sidebar.selectbox("Select Model", ["ResNet18", "ResNet50", "ResNet101", "MobileNetV2"])
    st.header("Detection Config")
    threshold = st.sidebar.slider("Threshold", 0.0, 1.0, 0.5, 0.05)
    st.header("About")

st.title("Driver Drowsiness Detection")

col1 = st.empty()

with col1:
    st.header("Input Image")
    if source_image:
        image = PIL.Image.open(source_image)
        st.image(image, use_column_width=True)
    else:
        st.write("Please upload an image.")
# with col2:
#     st.header("Input Video")
#     if source_video:
#         video = open(source_video, "rb").read()
#         st.video(video)
#     else:
#         st.write("Please upload a video.")

