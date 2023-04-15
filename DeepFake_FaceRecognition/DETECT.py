import argparse
import random

import cv2
import numpy as np
import torch

from DeepFake_FaceRecognition.FaceDetectionRecognition import infer_on_video_retina
from DeepFake_FaceRecognition.FaceDetectionRecognition import *
# # import sys
# # sys.path.append("./SelfBlendedImages-master")
# import DeepFake_FaceRecognition.SelfBlendedImages-master.src.inference.inference_video
import DeepFake_FaceRecognition.SelfBlendedImages.src.inference.inference_video
from DeepFake_FaceRecognition.SelfBlendedImages.src.inference import inference_video
from DeepFake_FaceRecognition.SelfBlendedImages.src.inference.preprocess import extract_frames
from DeepFake_FaceRecognition.LiveAdv.AdvDetection import views as Adv

from DeepFake_FaceRecognition.LiveAdv.LiveDetection import MtcnnLiveDetect as Live
import time
from DeepFake_FaceRecognition.spilt import eval as yoldetect
from DeepFake_FaceRecognition.keys import keysdetect


def detect():
    seed = 1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda')

    parser = argparse.ArgumentParser()
    # parser.add_argument('-clin', dest='client_name', default="Liaoxin", type=str,help="to get client name")
    parser.add_argument('-w', dest='weight_name',
                        default="G:\DeepFake_FaceRecognition (3)\DeepFake_FaceRecognition\SelfBlendedImages/weights/FFraw.tar",
                        type=str)
    # client="Liaoxin"
    # parser.add_argument('-i', dest='input_video',default='F:/test/{}.mp4'.format(client), type=str)#C:/Users/Fairy/Desktop/2.mp4
    parser.add_argument('-n', dest='n_frames', default=1, type=int)
    parser.add_argument('-in', dest='input_videotry', default='G:/pytest/Liaoxin.mp4', type=str)
    parser.add_argument("--detector",
                        default=r"G:\DeepFake_FaceRecognition (3)\DeepFake_FaceRecognition\LiveAdv\LiveDetection\face_detector",
                        help='人脸检测器存放文件夹')
    parser.add_argument("--face_confidence", type=float, default=0.8, help='人脸检测器阈值')
    parser.add_argument('--model_path',
                        default=r'G:\DeepFake_FaceRecognition (3)\DeepFake_FaceRecognition\LiveAdv/LiveDetection/model/modelface.pt',
                        help='活体检测模型路径')
    parser.add_argument("--live_confidence", type=float, default=0.75, help='活体检测阈值')
    parser.add_argument("--video", default=True, help='视频录制')

    parser.add_argument("-f", "--file_name", help="video file name", default='filename', type=str)
    parser.add_argument("-s", "--save_name", help="output file name", default='recording', type=str)
    parser.add_argument('-th', '--threshold', help='threshold to decide identical faces', default=1.54, type=float)
    parser.add_argument("-u", "--update", help="whether perform update the facebank", action="store_true")
    parser.add_argument("-tta", "--tta", help="whether test time augmentation", action="store_true")
    parser.add_argument("-c", "--score", help="whether show the confidence score", action="store_true")
    parser.add_argument("-b", "--begin", help="from when to start detection(in seconds)", default=0, type=int)
    parser.add_argument("-d", "--duration", help="perform detection for how long(in seconds)", default=0, type=int)
    parser.add_argument("runserver", defult=True, help='视频录制')
    parser.add_argument("localhost:8000", default=True, help='视频录制')

    args = parser.parse_args()
    flag = 0
    client = input("请输入用户名:")
    start_time = time.time()
    input_video = 'results/{}.mp4'.format(client)
    # input_video='results/{}.mp4'.format(args.client_name)

    print("活体检测环节")
    camera, out = Live.main(args, client)
    camera.release()
    out.release()
    cv2.destroyAllWindows()
    with open("results/MtcnnliveDetect_data.txt", "r") as f:
        MtcnnliveDetect_data = f.readline()
        MtcnnliveDetect_data = float(MtcnnliveDetect_data)
        print(MtcnnliveDetect_data)
        if MtcnnliveDetect_data == 0:
            print("活体检测警告")
            flag = 1

    print("--------------------------------------------------------------------")
    # #
    # #对抗样本环节
    print("对抗样本环节")
    Adv.main(args, client)
    with open("results/AdvDetection_data.txt", "r") as f:
        AdvDetection_data = f.readline()
        AdvDetection_data = float(AdvDetection_data)
        # print(AdvDetection_data)
        if AdvDetection_data < 0:
            print("对抗样本警告")
            flag = 1
    print("--------------------------------------------------------------------")

    # Deepfake环节
    print("Deepfake环节")
    inference_video.main(args, input_video)

    # filename=args.input_video

    with open("results/deepfake_data.txt", "r") as f:
        deepfake_data = f.readline()
        deepfake_data = float(deepfake_data)
        print(deepfake_data)
        if deepfake_data > 0.6:
            print("Deepfake警告")
            flag = 1
    print("--------------------------------------------------------------------")

    # 人脸识别
    print("人脸识别与人像分割环节")
    if (1):
        infer_on_video_retina.facedetect(args, input_video)
        end_time = time.time()
        print("系统所用时间:" + str(end_time - start_time))
        # 人像分割
        yoldetect.detect(input_video, client)

    # 关键点检测
    keysdetect(client)



if __name__ == '__main__':
    detect()
    # seed = 1
    # random.seed(seed)
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    #
    # device = torch.device('cuda')
    #
    # parser = argparse.ArgumentParser()
    # # parser.add_argument('-clin', dest='client_name', default="Liaoxin", type=str,help="to get client name")
    # parser.add_argument('-w', dest='weight_name', default="SelfBlendedImages/weights/FFraw.tar",type=str)
    # # client="Liaoxin"
    # # parser.add_argument('-i', dest='input_video',default='F:/test/{}.mp4'.format(client), type=str)#C:/Users/Fairy/Desktop/2.mp4
    # parser.add_argument('-n', dest='n_frames', default=1, type=int)
    # parser.add_argument('-in', dest='input_videotry',default='F:/test/Liaoxin.mp4', type=str)
    # parser.add_argument("--detector", default=r"LiveAdv\LiveDetection\face_detector", help='人脸检测器存放文件夹')
    # parser.add_argument("--face_confidence", type=float, default=0.8, help='人脸检测器阈值')
    # parser.add_argument('--model_path', default=r'LiveAdv/LiveDetection/model/modelface.pt', help='活体检测模型路径')
    # parser.add_argument("--live_confidence", type=float, default=0.75, help='活体检测阈值')
    # parser.add_argument("--video", default=True, help='视频录制')
    #
    # parser.add_argument("-f", "--file_name", help="video file name", default='filename', type=str)
    # parser.add_argument("-s", "--save_name", help="output file name", default='recording', type=str)
    # parser.add_argument('-th', '--threshold', help='threshold to decide identical faces', default=1.54, type=float)
    # parser.add_argument("-u", "--update", help="whether perform update the facebank", action="store_true")
    # parser.add_argument("-tta", "--tta", help="whether test time augmentation", action="store_true")
    # parser.add_argument("-c", "--score", help="whether show the confidence score", action="store_true")
    # parser.add_argument("-b", "--begin", help="from when to start detection(in seconds)", default=0, type=int)
    # parser.add_argument("-d", "--duration", help="perform detection for how long(in seconds)", default=0, type=int)
    #
    #
    #
    # args = parser.parse_args()
    # flag=0
    # client=input("请输入用户名:")
    # start_time = time.time()
    # input_video='results/{}.mp4'.format(client)
    # # input_video='results/{}.mp4'.format(args.client_name)
    #
    #
    # print("活体检测环节")
    # camera, out = Live.main(args,client)
    # camera.release()
    # out.release()
    # cv2.destroyAllWindows()
    # with open("results/MtcnnliveDetect_data.txt", "r") as f:
    #     MtcnnliveDetect_data = f.readline()
    #     MtcnnliveDetect_data = float(MtcnnliveDetect_data)
    #     print(MtcnnliveDetect_data)
    #     if MtcnnliveDetect_data == 0:
    #         print("活体检测警告")
    #         flag = 1
    #
    # print("--------------------------------------------------------------------")
    # # #
    # # #对抗样本环节
    # print("对抗样本环节")
    # Adv.main(args,client)
    # with open("results/AdvDetection_data.txt", "r") as f:
    #     AdvDetection_data = f.readline()
    #     AdvDetection_data = float(AdvDetection_data)
    #     # print(AdvDetection_data)
    #     if AdvDetection_data < 0:
    #         print("对抗样本警告")
    #         flag=1
    # print("--------------------------------------------------------------------")
    #
    #
    # #Deepfake环节
    # print("Deepfake环节")
    # inference_video.main(args,input_video)
    #
    #
    # #filename=args.input_video
    #
    #
    # with open("results/deepfake_data.txt", "r") as f:
    #     deepfake_data = f.readline()
    #     deepfake_data = float(deepfake_data)
    #     print(deepfake_data)
    #     if deepfake_data>0.6:
    #         print("Deepfake警告")
    #         flag=1
    # print("--------------------------------------------------------------------")
    #
    #
    # #人脸识别
    # print("人脸识别与人像分割环节")
    # if(1):
    #     infer_on_video_retina.facedetect(args,input_video)
    #     end_time = time.time()
    #     print("系统所用时间:"+str(end_time-start_time))
    # # 人像分割
    #     yoldetect.detect(input_video,client)
