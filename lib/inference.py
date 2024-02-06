import tensorflow as tf
import cv2
import numpy as np
import sys
sys.path.insert(0, 'FaceBoxesV2')
sys.path.insert(0, '..')
from faceboxes_detector import *
import torch
import PIL.Image as Image
import torchvision.transforms as transforms
from networks import *
import data_utils
from functions import *

det_head = 'pip'
net_stride = 32
batch_size = 16
init_lr = 0.0001
num_epochs = 200
decay_steps = [30, 50]
input_size = 256
backbone = 'resnet18'
pretrained = True
criterion_cls = 'l2'
criterion_reg = 'l1'
cls_loss_weight = 10
reg_loss_weight = 1
num_lms = 16
save_interval = num_epochs
num_nb = 10
use_gpu = True
gpu_id = 0
meanface_indices, reverse_index1, reverse_index2, max_len = get_meanface(os.path.join('data', "WFLW", 'meanface.txt'), num_nb)

def forward_pip_tflite(net, inputs, preprocess, input_size, net_stride, num_nb):
    # Load the TFLite model and allocate tensors
    # interpreter = tf.lite.Interpreter(model_path=net)
    # interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], inputs)

    # Run the model
    interpreter.invoke()

    # Get the output tensors
    outputs_cls, outputs_x, outputs_y, outputs_nb_x, outputs_nb_y = [interpreter.get_tensor(output['index']) for output in output_details]

    tmp_batch, tmp_channel, tmp_height, tmp_width = outputs_cls.shape
    assert tmp_batch == 1

    outputs_cls = np.reshape(outputs_cls, (tmp_batch*tmp_channel, -1))
    max_ids = np.argmax(outputs_cls, axis=1)
    max_cls = np.max(outputs_cls, axis=1)
    max_ids = np.reshape(max_ids, (-1, 1))
    max_ids_nb = np.repeat(max_ids, num_nb, axis=1).reshape(-1, 1)

    outputs_x = np.reshape(outputs_x, (tmp_batch*tmp_channel, -1))
    outputs_x_select = np.take_along_axis(outputs_x, max_ids, axis=1).squeeze(1)
    outputs_y = np.reshape(outputs_y, (tmp_batch*tmp_channel, -1))
    outputs_y_select = np.take_along_axis(outputs_y, max_ids, axis=1).squeeze(1)

    outputs_nb_x = np.reshape(outputs_nb_x, (tmp_batch*num_nb*tmp_channel, -1))
    outputs_nb_x_select = np.take_along_axis(outputs_nb_x, max_ids_nb, axis=1).squeeze(1).reshape(-1, num_nb)
    outputs_nb_y = np.reshape(outputs_nb_y, (tmp_batch*num_nb*tmp_channel, -1))
    outputs_nb_y_select = np.take_along_axis(outputs_nb_y, max_ids_nb, axis=1).squeeze(1).reshape(-1, num_nb)

    tmp_x = (max_ids%tmp_width).reshape(-1,1).astype(float)+outputs_x_select.reshape(-1,1)
    tmp_y = (max_ids//tmp_width).reshape(-1,1).astype(float)+outputs_y_select.reshape(-1,1)
    tmp_x /= 1.0 * input_size / net_stride
    tmp_y /= 1.0 * input_size / net_stride

    tmp_nb_x = (max_ids%tmp_width).reshape(-1,1).astype(float)+outputs_nb_x_select
    tmp_nb_y = (max_ids//tmp_width).reshape(-1,1).astype(float)+outputs_nb_y_select
    tmp_nb_x = tmp_nb_x.reshape(-1, num_nb)
    tmp_nb_y = tmp_nb_y.reshape(-1, num_nb)
    tmp_nb_x /= 1.0 * input_size / net_stride
    tmp_nb_y /= 1.0 * input_size / net_stride

    return tmp_x, tmp_y, tmp_nb_x, tmp_nb_y, outputs_cls, max_cls


# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="/home/kyv/WD_500G/Project/PIPNet/results/new_model_299.tflite")
interpreter.allocate_tensors()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

detector = FaceBoxesDetector('FaceBoxes', 'FaceBoxesV2/weights/FaceBoxesV2.pth', True, device)


# Get input and output tensors
input_tensor_index = interpreter.get_input_details()[0]['index']
output = interpreter.tensor(interpreter.get_output_details()[0]['index'])
video_file = '/dev/video2'
# Open a video file or camera stream
if video_file == 'camera':
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(video_file)
if (cap.isOpened()== False): 
    print("Error opening video stream or file")
    
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor(), normalize])


# Loop through video frames
while cap.isOpened():
    ret, frame = cap.read()
    my_thresh = 0.9
    det_box_scale = 1.2

    if ret == True:
        detections, _ = detector.detect(frame, my_thresh, 1)
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
            det_xmax = min(det_xmax, frame_width-1)
            det_ymax = min(det_ymax, frame_height-1)
            det_width = det_xmax - det_xmin + 1
            det_height = det_ymax - det_ymin + 1
            cv2.rectangle(frame, (det_xmin, det_ymin), (det_xmax, det_ymax), (0, 0, 255), 2)
            det_crop = frame[det_ymin:det_ymax, det_xmin:det_xmax, :]
            det_crop = cv2.resize(det_crop, (input_size, input_size))
            inputs = Image.fromarray(det_crop[:,:,::-1].astype('uint8'), 'RGB')
            inputs = preprocess(inputs).unsqueeze(0)
            inputs = inputs.to(device)
            interpreter_path = "/home/kyv/WD_500G/Project/PIPNet/results/new_model_299.tflite"
            lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = forward_pip_tflite(interpreter_path, inputs, preprocess, input_size, net_stride, num_nb)
            lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=1).flatten()
            tmp_nb_x = lms_pred_nb_x[reverse_index1, reverse_index2].view(num_lms, max_len)
            tmp_nb_y = lms_pred_nb_y[reverse_index1, reverse_index2].view(num_lms, max_len)
            tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1,1)
            tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1,1)
            lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten()
            lms_pred = lms_pred.cpu().numpy()
            lms_pred_merge = lms_pred_merge.cpu().numpy()
            for i in range(num_lms):
                x_pred = lms_pred_merge[i*2] * det_width
                y_pred = lms_pred_merge[i*2+1] * det_height
                cv2.circle(frame, (int(x_pred)+det_xmin, int(y_pred)+det_ymin), 1, (0, 0, 255), -1)
            

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
