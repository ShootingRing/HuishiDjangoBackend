import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms,models,utils
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image
import sys
import random
import shutil
from model import Detector
import argparse
from datetime import datetime
from tqdm import tqdm
from retinaface.pre_trained_models import get_model
from DeepFake_FaceRecognition.SelfBlendedImages.src.inference.preprocess import extract_frames
import warnings
warnings.filterwarnings('ignore')

def main(args,input):
    device = torch.device('cuda')
    model=Detector()
    model=model.to(device)
    cnn_sd=torch.load(args.weight_name)["model"]
    model.load_state_dict(cnn_sd)
    model.eval()

    face_detector = get_model("resnet50_2020-07-20", max_size=2048,device=device)
    face_detector.eval()

    face_list,idx_list=extract_frames(input,args.n_frames,face_detector)

    with torch.no_grad():
        img=torch.tensor(face_list).to(device).float()/255
        pred=model(img).softmax(1)[:,1]
        
        
    pred_list=[]
    idx_img=-1
    for i in range(len(pred)):
        if idx_list[i]!=idx_img:
            pred_list.append([])
            idx_img=idx_list[i]
        pred_list[-1].append(pred[i].item())
    pred_res=np.zeros(len(pred_list))
    for i in range(len(pred_res)):
        pred_res[i]=max(pred_list[i])
        print(pred_res[i])
    pred=pred_res.mean()

    print(f'fakeness: {pred:.4f}')
    with open("results/deepfake_data.txt","w") as f:
        f.write(str(pred))  # 自带文件关闭功能，不需要再写f.close()





if __name__=='__main__':

    seed=1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda')

    parser=argparse.ArgumentParser()
    parser.add_argument('-w',dest='weight_name',default="weights/FFraw.tar",type=str)
    parser.add_argument('-i',dest='input_video',default="C:/Users/Fairy/Desktop/1.mp4",type=str)
    parser.add_argument('-n',dest='n_frames',default=32,type=int)
    args=parser.parse_args()

    main(args)

