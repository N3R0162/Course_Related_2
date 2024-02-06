import cv2, os
import sys
sys.path.insert(0, '..')
import numpy as np
from PIL import Image
import logging
import pickle
import importlib
from math import floor
import time

import torch
import torch.nn as nn
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


from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt

def aspect_ratio_calculation(lms_pred):
    point_0 = np.array([lms_pred[0], lms_pred[1]])
    point_1 = np.array([lms_pred[2], lms_pred[3]])
    point_2 = np.array([lms_pred[4], lms_pred[5]])
    point_3 = np.array([lms_pred[6], lms_pred[7]])
    point_4 = np.array([lms_pred[8], lms_pred[9]])
    point_5 = np.array([lms_pred[10], lms_pred[11]])
    point_6 = np.array([lms_pred[12], lms_pred[13]])
    point_7 = np.array([lms_pred[14], lms_pred[15]])
    point_8 = np.array([lms_pred[16], lms_pred[17]])
    point_9 = np.array([lms_pred[18], lms_pred[19]])
    point_10 = np.array([lms_pred[20], lms_pred[21]])
    point_11 = np.array([lms_pred[22], lms_pred[23]])
    point_12 = np.array([lms_pred[24], lms_pred[25]])
    point_13 = np.array([lms_pred[26], lms_pred[27]])
    point_14 = np.array([lms_pred[28], lms_pred[29]])
    point_15 = np.array([lms_pred[30], lms_pred[31]])


    left_height_1 = np.linalg.norm(point_1 - point_7)
    left_height_2 = np.linalg.norm(point_2 - point_6)
    left_height_3 = np.linalg.norm(point_3 - point_5)
    left_width = np.linalg.norm(point_0 - point_4)
    left_eye_aspect_ratio = (left_height_1 + left_height_2 + left_height_3) / (3 * left_width)

    right_height_1 = np.linalg.norm(point_9 - point_15)
    right_height_2 = np.linalg.norm(point_10 - point_14)    
    right_height_3 = np.linalg.norm(point_11 - point_13)
    right_width = np.linalg.norm(point_8 - point_12)
    right_eye_aspect_ratio = (right_height_1 + right_height_2 + right_height_3) / (3 * right_width)

    average_aspect_ratio = (left_eye_aspect_ratio + right_eye_aspect_ratio) / 2
    return average_aspect_ratio

#Check the sh input to see if it has enough argument:
if not len(sys.argv) == 4:
    print('Format:')
    print('python lib/test.py config_file test_labels test_images')
    exit(0)
experiment_name = sys.argv[1].split('/')[-1][:-3]
data_name = sys.argv[1].split('/')[-2]
test_labels = sys.argv[2]
test_images = sys.argv[3]
config_path = '.experiments.{}.{}'.format(data_name, experiment_name)


#Get and modify values from config file
my_config = importlib.import_module(config_path, package='PIPNet')
Config = getattr(my_config, 'Config')
cfg = Config()
cfg.experiment_name = experiment_name
cfg.data_name = data_name

os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu_id)

if not os.path.exists(os.path.join('./logs', cfg.data_name)):
    os.mkdir(os.path.join('./logs', cfg.data_name))
log_dir = os.path.join('./logs', cfg.data_name, cfg.experiment_name)
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

logging.basicConfig(filename=os.path.join(log_dir, 'train.log'), level=logging.INFO)

save_dir = os.path.join('./snapshots', cfg.data_name, cfg.experiment_name)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

if cfg.det_head == 'pip':
    meanface_indices, reverse_index1, reverse_index2, max_len = get_meanface(os.path.join('data', cfg.data_name, 'meanface.txt'), cfg.num_nb)
    
if cfg.det_head == 'pip':
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
else:
    print('No such head:', cfg.det_head)
    exit(0)

if cfg.use_gpu:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
net = net.to(device)

weight_file = os.path.join(save_dir, 'epoch%d.pth' % (cfg.num_epochs-1))
# weight_file = os.path.join(save_dir, 'epoch4.pth')
state_dict = torch.load(weight_file, map_location=torch.device("cpu"))
net.load_state_dict(state_dict)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([transforms.Resize((cfg.input_size, cfg.input_size)), transforms.ToTensor(), normalize])

norm_indices = None
if cfg.data_name == 'data_300W' or cfg.data_name == 'data_300W_COFW_WFLW' or cfg.data_name == 'data_300W_CELEBA':
    norm_indices = [36, 45]
elif cfg.data_name == 'COFW':
    norm_indices = [8, 9]
elif cfg.data_name == 'WFLW':
    norm_indices = [0, 15]
elif cfg.data_name == 'AFLW':
    pass
elif cfg.data_name == 'LaPa':
    norm_indices = [0, 15]
elif cfg.data_name == "Iris":
    norm_indices = [0, 4]
else:
    print('No such data!')
    exit(0)

labels = get_label(cfg.data_name, test_labels)

nmes_std = []
nmes_merge = []
norm = None
time_all = 0

from tqdm import tqdm
for label in tqdm(labels):
    image_name = label[0]
    lms_gt = label[1]
    # print("********************************************************")
    # print(image_name)
    # print(lms_gt)
    # print("********************************************************")
    # exit()
    if cfg.data_name == 'data_300W' or cfg.data_name == 'COFW' or cfg.data_name == 'WFLW' or cfg.data_name == 'data_300W_COFW_WFLW' or cfg.data_name == 'data_300W_CELEBA' or cfg.data_name == "LaPa" or cfg.data_name == "Iris":
        norm = np.linalg.norm(lms_gt.reshape(-1, 2)[norm_indices[0]] - lms_gt.reshape(-1, 2)[norm_indices[1]])
    elif cfg.data_name == 'AFLW':
        norm = 1
    image_path = os.path.join('data', cfg.data_name, test_images, image_name)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (cfg.input_size, cfg.input_size))
    inputs = Image.fromarray(image[:,:,::-1].astype('uint8'), 'RGB')
    inputs = preprocess(inputs).unsqueeze(0)
    inputs = inputs.to(device)
    t1 = time.time()
    if cfg.det_head == 'pip':
        lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = forward_pip(net, inputs, preprocess, cfg.input_size, cfg.net_stride, cfg.num_nb)
        predicted_scores = outputs_cls.cpu().numpy()
    else:
        print('No such head!')
    if cfg.det_head == 'pip':
        # merge neighbor predictions
        lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=1).flatten()
        tmp_nb_x = lms_pred_nb_x[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
        tmp_nb_y = lms_pred_nb_y[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
        tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1,1)
        tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1,1)
        lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten()
    t2 = time.time()
    time_all += (t2-t1)

    if cfg.det_head == 'pip':
        lms_pred = lms_pred.cpu().numpy()
        lms_pred_merge = lms_pred_merge.cpu().numpy()

    #Print picture name and predicted landmarks:
    # print(" ")
    # print("Image name: ", image_name)
    # print("LMS PRED: ", lms_pred_merge)
    # aspect_ratio = aspect_ratio_calculation(lms_pred_merge)

    # # Visualize: 
    # landmarks_pred = lms_pred

    # for i in range(0, len(landmarks_pred), 2):
    #     x = int(landmarks_pred[i] * cfg.input_size)
    #     y = int(landmarks_pred[i + 1] * cfg.input_size)
    #     cv2.circle(image, (x, y), radius=1, color=(0, 0, 255), thickness=2)
    
    # # Convert the aspect ratio to a string
    # aspect_ratio_text = "Aspect Ratio: {:.2f}".format(aspect_ratio)

    # # Draw the aspect ratio on the image
    # cv2.putText(image, aspect_ratio_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # cv2.imshow("Pred", image)
    
    # while True:
    #     key = cv2.waitKey(1)
    #     if key == 27 or cv2.getWindowProperty("Pred", cv2.WND_PROP_VISIBLE) < 1:
    #         break

    nme_std = compute_nme(lms_pred, lms_gt, norm)
    nmes_std.append(nme_std)
    if cfg.det_head == 'pip':
        nme_merge = compute_nme(lms_pred_merge, lms_gt, norm)
        nmes_merge.append(nme_merge)


# print('Total inference time:', time_all)
# print('Image num:', len(labels))
# print('Average inference time:', time_all/len(labels))

if cfg.det_head == 'pip':
    print('nme: {}'.format(np.mean(nmes_merge)))
    logging.info('nme: {}'.format(np.mean(nmes_merge)))

    fr, auc_value = compute_fr_and_auc(nmes_merge)
    print('fr : {}'.format(fr))
    logging.info('fr : {}'.format(fr))
    print('auc: {}'.format(auc_value))
    logging.info('auc: {}'.format(auc_value))



