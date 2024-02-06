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
from tqdm import tqdm
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
from mobilenetv3 import mobilenetv3_large

if not len(sys.argv) == 4:
    print('Format:')
    print('python lib/demo.py config_file image_file')
    exit(0)
experiment_name = sys.argv[1].split('/')[-1][:-3]
data_name = sys.argv[1].split('/')[-2]
config_path = '.experiments.{}.{}'.format(data_name, experiment_name)
image_path = sys.argv[2]

my_config = importlib.import_module(config_path, package='PIPNet')
Config = getattr(my_config, 'Config')
cfg = Config()
cfg.experiment_name = experiment_name
cfg.data_name = data_name

save_dir = sys.argv[3]

meanface_indices, reverse_index1, reverse_index2, max_len = get_meanface(os.path.join('data', cfg.data_name, 'meanface.txt'), cfg.num_nb)

if cfg.backbone == 'resnet18':
    resnet18 = models.resnet18(pretrained=cfg.pretrained)
    net = Pip_resnet18(resnet18, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
elif cfg.backbone == 'resnet50':
    resnet50 = models.resnet50(pretrained=cfg.pretrained)
    net = Pip_resnet50(resnet50, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
elif cfg.backbone == 'resnet101':
    resnet101 = models.resnet101(pretrained=cfg.pretrained)
    net = Pip_resnet101(resnet101, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
elif cfg.backbone == 'mobilenet_v2':
    mbnet = models.mobilenet_v2(pretrained=cfg.pretrained)
    net = Pip_mbnetv2(mbnet, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
elif cfg.backbone == 'mobilenet_v3':
    mbnet = mobilenetv3_large()
    if cfg.pretrained:
        mbnet.load_state_dict(torch.load('lib/mobilenetv3-large-1cd25616.pth'))
    net = Pip_mbnetv3(mbnet, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
else:
    print('No such backbone!')
    exit(0)

if cfg.use_gpu:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
net = net.to(device)

weight_file = os.path.join(save_dir, 'epoch%d.pth' % (cfg.num_epochs-1))
state_dict = torch.load(weight_file, map_location=device)
net.load_state_dict(state_dict)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([transforms.Resize((cfg.input_size, cfg.input_size)), transforms.ToTensor(), normalize])

def demo_image(image_path, net, preprocess, input_size, net_stride, num_nb, use_gpu, device):
    detector = FaceBoxesDetector('FaceBoxes', 'FaceBoxesV2/weights/FaceBoxesV2.pth', use_gpu, device)
    my_thresh = 0.6
    det_box_scale = 1.2
    bbox_list = []

    net.eval()
    for image_file in tqdm(os.listdir(image_path)):
        file_path = os.path.join(image_path, image_file)
        image = cv2.imread(file_path)
        image_path_new = file_path.split('/')
        image_name = image_path_new[-2]+"/"+image_path_new[-1]
        image_height, image_width, _ = image.shape
        detections, _ = detector.detect(image, my_thresh, 1)
        # Check if decetions is empty, and add to a txt file if it is empty
        if len(detections) == 0:
            current_dir = os.getcwd()
            relative_path = "data/Inference/image_annotation"
            destination_folder = os.path.join(current_dir, relative_path)
            if destination_folder is not None and not os.path.exists(destination_folder):
                os.makedirs(destination_folder)
            with open(os.path.join(destination_folder, 'nofacebox.txt'), 'a') as f:
                #Turn bbox from list of list to line of number separate by comma and write to the f with the image_name with original la
                f.write(image_name + ' ' + ' '.join(map(str, [0,0,0,0])) + '\n')
                
        for i in range(len(detections)):
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
            det_xmax = min(det_xmax, image_width-1)
            det_ymax = min(det_ymax, image_height-1)
            det_width = det_xmax - det_xmin + 1
            det_height = det_ymax - det_ymin + 1
            # cv2.rectangle(image, (det_xmin, det_ymin), (det_xmax, det_ymax), (0, 0, 255), 2)
            det_crop = image[det_ymin:det_ymax, det_xmin:det_xmax, :]
            det_crop = cv2.resize(det_crop, (input_size, input_size))
            inputs = Image.fromarray(det_crop[:,:,::-1].astype('uint8'), 'RGB')
            inputs = preprocess(inputs).unsqueeze(0)
            inputs = inputs.to(device)
            lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = forward_pip(net, inputs, preprocess, input_size, net_stride, num_nb)
            lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=1).flatten()
            tmp_nb_x = lms_pred_nb_x[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
            tmp_nb_y = lms_pred_nb_y[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
            tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1,1)
            tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1,1)
            lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten()
            lms_pred = lms_pred.cpu().numpy()
            lms_pred_merge = lms_pred_merge.cpu().numpy()
            bbox = [det_xmin, det_ymin, det_xmax, det_ymax]
            cur_dir = os.getcwd()
            relative_path = "data/Inference/image_annotation"
            destination_folder = os.path.join(cur_dir, relative_path)
            if destination_folder is not None and not os.path.exists(destination_folder):
                os.makedirs(destination_folder)
            with open(os.path.join(destination_folder, 'annotation.txt'), 'a') as f:
                #Turn bbox from list of list to line of number separate by comma and write to the f with the image_name:
                f.write(image_name + ' ' + ' '.join(map(str, bbox)) + '\n')


        current_dir = os.getcwd()
    
        # Specify the relative path to the destination folder
        relative_path = "data/Inference/image_train"

        # Combine the current directory and relative path
        destination_folder = os.path.join(current_dir, relative_path)
        if destination_folder is not None and not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        # Update the image_crop_name with the new destination folder path
        image_crop_name = os.path.join(destination_folder, image_file)
        cv2.imwrite(image_crop_name, det_crop)
        # cv2.imshow("Facial Landmarks", image)
        # while True:
        #     key = cv2.waitKey(1)    
            
        #     # Check if the "Esc" key is pressed or the window is closed
        #     if key == 27 or cv2.getWindowProperty('Facial Landmarks', cv2.WND_PROP_VISIBLE) < 1:
        #         break

    print(bbox_list)
demo_image(image_path, net, preprocess, cfg.input_size, cfg.net_stride, cfg.num_nb, cfg.use_gpu, device)
