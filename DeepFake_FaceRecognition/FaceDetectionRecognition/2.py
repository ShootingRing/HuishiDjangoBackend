import argparse
from pathlib import Path
from PIL import Image
from mtcnn import MTCNN
from datetime import datetime

from PIL import Image
import numpy as np
from mtcnn_pytorch.src.align_trans import get_reference_facial_points, warp_and_crop_face

##LC
import cv2
import torch
from models.retinaface import RetinaFace
from data.config import cfg_mnet, cfg_re50
from convert_to_onnx import load_model
from utilsRe.timer import Timer
import torch.backends.cudnn as cudnn
from layers.functions.prior_box import PriorBox
from utilsRe.box_utils import decode, decode_landm
from utilsRe.nms.py_cpu_nms import py_cpu_nms
##LC

def take_save_pic(det_model_path, save_path):
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    # 摄像头默认像素640*480，可以根据摄像头素质调整分辨率
    cap.set(3,1280)
    cap.set(4,720)

    if 'Resnet' in det_model_path:
        cfg = cfg_re50
    else:
        cfg = cfg_mnet

    trained_model = det_model_path#'./weights/mobilenet0.25_Final.pth'
    cpu = False
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, trained_model, cpu)
    net.eval()
    cudnn.benchmark = True
    device = torch.device("cpu" if cpu else "cuda")
    net = net.to(device)
    idx = 1

    while cap.isOpened():
        # 采集一帧一帧的图像数据
        isSuccess,frame = cap.read()
        # 实时的将采集到的数据显示到界面上
        if isSuccess:
            # frame_text = cv2.putText(frame,
            #             'Press t to take a picture,q to quit.....',
            #             (10,100),
            #             cv2.FONT_HERSHEY_SIMPLEX,
            #             2,
            #             (0,255,0),
            #             3,
            #             cv2.LINE_AA)
            # cv2.imshow("My Capture",frame_text)
            cv2.imshow("My Capture", frame)
        # 实现按下“t”键拍照
        if cv2.waitKey(1)&0xFF == ord('t'):

            print(save_path)
            warped_face = np.array(net.align(frame))[...,::-1]

            cv2.imwrite(str(save_path / '{}.jpg'.format(name + str(idx))), warped_face)
            idx += 1

        if cv2.waitKey(1)&0xFF == ord('q'):
            break

    # 释放摄像头资源
    cap.release()

def get_args():
    parser = argparse.ArgumentParser(description='take a picture')
    parser.add_argument('--name','-n', default='unknown', type=str,help='input the name of the recording person')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    #args = get_args()
    data_path = Path('data')
    name = 'liu'#args.name
    save_path = data_path/'facebank/single'/name
    if not save_path.exists():
        save_path.mkdir()

    det_model_path = './FaceDetectionRecognition/weights/Resnet50_Final.pth'
    image = cv2.imread("D:/QQ接收文件/1091714334/Image/C2C/DFD6927E4B83E94749250AA3ED8FF94E.jpg")
    if 'Resnet' in det_model_path:
        cfg = cfg_re50
    else:
        cfg = cfg_mnet

    trained_model = det_model_path  # './weights/mobilenet0.25_Final.pth'
    cpu = False
    net = RetinaFace(cfg=cfg, phase='test')
    net = load_model(net, trained_model, cpu)
    net.eval()
    cudnn.benchmark = True
    device = torch.device("cpu" if cpu else "cuda")
    net = net.to(device)
    warped_face = np.array(net.align( image ))[...,::-1]

    cv2.imwrite("try", warped_face)