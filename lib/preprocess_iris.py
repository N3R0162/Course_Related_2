import os, cv2
import hdf5storage
import numpy as np
import sys
from tqdm import tqdm

def gen_meanface(root_folder, data_name):
    with open(os.path.join(root_folder, data_name, 'train.txt'), 'r') as f:
        annos = f.readlines()
    annos = [x.strip().split()[1:] for x in annos]
    annos = [[float(x.rstrip(',')) for x in anno] for anno in annos]    
    annos = np.array(annos)
    meanface = np.mean(annos, axis=0)
    meanface = meanface.tolist()
    meanface = [str(x) for x in meanface]
    path = os.path.join(root_folder, data_name, 'meanface.txt')
    print(path)
    with open(os.path.join(root_folder, data_name, 'meanface.txt'), 'w') as f:
        f.write(' '.join(meanface))


gen_meanface('/home/kyv/Desktop/PIPNet/data', 'Iris')
