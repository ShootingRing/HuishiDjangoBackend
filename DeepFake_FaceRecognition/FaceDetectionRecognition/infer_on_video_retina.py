# import cv2
# from PIL import Image
# import argparse
# from pathlib import Path
# import torch
#
# from mtcnn import MTCNN
# from Learner import face_learner
# from utils import load_facebank, draw_box_name, prepare_facebank
# from models.retinaface import RetinaFace
import cv2
from PIL import Image
import argparse
from pathlib import Path
# from multiprocessing import Process, Pipe,Value,Array
import torch
from DeepFake_FaceRecognition.FaceDetectionRecognition.config import get_config
#from mtcnn import MTCNN
from DeepFake_FaceRecognition.FaceDetectionRecognition.Learner import face_learner
from DeepFake_FaceRecognition.FaceDetectionRecognition.utils import load_facebank, draw_box_name, prepare_facebank

from DeepFake_FaceRecognition.FaceDetectionRecognition.mtcnn_pytorch.src.align_trans import get_reference_facial_points,warp_and_crop_face
##
from DeepFake_FaceRecognition.FaceDetectionRecognition.models.retinaface import RetinaFace
from DeepFake_FaceRecognition.FaceDetectionRecognition.data.config import cfg_mnet, cfg_re50
from DeepFake_FaceRecognition.FaceDetectionRecognition.convert_to_onnx import load_model
from DeepFake_FaceRecognition.FaceDetectionRecognition.utilsRe.timer import Timer
import numpy as np
import torch.backends.cudnn as cudnn
from DeepFake_FaceRecognition.FaceDetectionRecognition.layers.functions.prior_box import PriorBox
from DeepFake_FaceRecognition.FaceDetectionRecognition.utilsRe.box_utils import decode, decode_landm
from DeepFake_FaceRecognition.FaceDetectionRecognition.utilsRe.nms.py_cpu_nms import py_cpu_nms
from DeepFake_FaceRecognition.FaceDetectionRecognition.utils import prepare_facebank_retina

def Mylign_multi(img, detRes, vis_thres):
    boxes = []
    landmarks = []
    for b in detRes:
        if b[4] < vis_thres:
            continue
        box = [b[0], b[1],b[2], b[3], b[4]]
        landmark = [b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14]]
        boxes.append(box)
        landmarks.append(landmark)

    faces = []
    refrence = get_reference_facial_points(default_square= True)
    for landmark in landmarks:
        facial5points = []
        ij = 0
        for j in range(5):
            l1 = [landmark[ij],landmark[ij+1]]
            facial5points.append(l1)
            ij += 2
        #facial5points = [[landmark[j],landmark[j+SelfBlendedImages]] for j in range(5)]
        warped_face = warp_and_crop_face(np.array(img), facial5points, refrence, crop_size=(112,112))
        # width=112
        # warped_face1 = cv2.resize(warped_face, (width, width)).astype(np.float32)
        cv2.imshow('warped_face Capture',  cv2.resize(warped_face,(224,224) ))
        faces.append(Image.fromarray(warped_face))
    return boxes, landmarks, faces

# if __name__ == '__main__':
def facedetect(args,filename):
    # with open("F:/deepfake_data.txt", "r") as f:
    #     deepfake_data=f.readline()
    #     deepfake_data=float(deepfake_data)
    #     print(deepfake_data)
    # with open("F:/AdvDetection_data.txt", "r") as f:
    #     AdvDetection_data = f.readline()
    #     AdvDetection_data = float(AdvDetection_data)
    #     print(AdvDetection_data)
    # with open("F:/MtcnnliveDetect_data.txt", "r") as f:
    #     MtcnnliveDetect_data = f.readline()
    #     MtcnnliveDetect_data = float(MtcnnliveDetect_data)
    #     print(MtcnnliveDetect_data)
    #if(deepfake_data<0.6):
    # parser = argparse.ArgumentParser(description='for face verification')
    # parser.add_argument("-f", "--file_name", help="video file name", default='filename', type=str)
    # parser.add_argument("-s", "--save_name", help="output file name", default='recording', type=str)
    # parser.add_argument('-th', '--threshold', help='threshold to decide identical faces', default=1.54, type=float)
    # parser.add_argument("-u", "--update", help="whether perform update the facebank", action="store_true")
    # parser.add_argument("-tta", "--tta", help="whether test time augmentation", action="store_true")
    # parser.add_argument("-c", "--score", help="whether show the confidence score", action="store_true")
    # parser.add_argument("-b", "--begin", help="from when to start detection(in seconds)", default=0, type=int)
    # parser.add_argument("-d", "--duration", help="perform detection for how long(in seconds)", default=0, type=int)
    bank_name = 'MyBank'
    # args = parser.parse_args()
    conf = get_config(False)
    # conf.facebank_path = conf.data_path / 'facebank/' / Path(bank_name + '/')
    conf.facebank_path = conf.data_path / 'facebank/' / Path('MyBank' + '/')
    # print(conf.facebank_path)
    torch.set_grad_enabled(False)
    confidence_threshold = 0.1
    nms_threshold = 0.4
    vis_thres = 0.6
    cpu = False
    det_model_path = 'G:\DeepFake_FaceRecognition (3)\DeepFake_FaceRecognition\FaceDetectionRecognition/weights/Resnet50_Final.pth'
    if 'Resnet' in det_model_path:
        cfg = cfg_re50
    else:
        cfg = cfg_mnet

    net = RetinaFace(cfg=cfg, phase='test')
    trained_model = det_model_path
    net = load_model(net, trained_model, cpu)
    net.eval()
    # print('Finished loading model!')
    # print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if cpu else "cuda")
    net = net.to(device)
    resize = 1
    _t = {'forward_pass': Timer(), 'misc': Timer()}
    ##LC

    det_model_path = 'G:\DeepFake_FaceRecognition (3)\DeepFake_FaceRecognition\FaceDetectionRecognition/weights/Resnet50_Final.pth'
    cpu = False
    resize = 1
    _t = {'forward_pass': Timer(), 'misc': Timer()}
    device = torch.device("cpu" if cpu else "cuda")
    if 'Resnet' in det_model_path:
        cfg = cfg_re50
    else:
        cfg = cfg_mnet
    net = RetinaFace(cfg=cfg, phase='test')
    trained_model = det_model_path
    net = load_model(net, trained_model, cpu)
    net = net.to('cuda')
    net.eval()
    # print('Finished loading RetinaFace model!')
    # print(net)
    confidence_threshold = 0.1
    nms_threshold = 0.4
    vis_thres = 0.6
    learner = face_learner(conf, True)
    learner.threshold = args.threshold
    if conf.device.type == 'cpu':
        learner.load_state(conf, 'cpu_final.pth', True, True)
    else:
        learner.load_state(conf, 'final.pth', True, True)
    learner.model.to("cuda")
    learner.model.eval()
    # print('learner loaded')
    # print(learner.model)
    update = False
    if update:
        targets, names = prepare_facebank_retina(conf, learner.model, net, tta=args.tta)
        # print('facebank updated')
    else:
        targets, names = load_facebank(conf)
        # print('facebank loaded')
    # str(conf.facebank_path / args.file_name)
    cap = cv2.VideoCapture(filename)

    cap.set(cv2.CAP_PROP_POS_MSEC, args.begin * 1000)

    fps = cap.get(cv2.CAP_PROP_FPS)

    video_writer = cv2.VideoWriter(str(conf.facebank_path / '{}.avi'.format(args.save_name)),
                                   cv2.VideoWriter_fourcc(*'mp4v'), int(fps), (1280, 720))

    if args.duration != 0:
        i = 0
    print(cap.isOpened())

    while cap.isOpened():
        isSuccess, frame = cap.read()
        # print(isSuccess)
        if isSuccess:
            img_raw = frame
            image = Image.fromarray(frame)
            img = np.float32(frame)
            im_height, im_width, _ = img.shape
            scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            img -= (104, 117, 123)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.to(device)
            scale = scale.to(device)
            _t['forward_pass'].tic()
            loc, conf, landms = net(img)  # forward pass
            _t['forward_pass'].toc()
            _t['misc'].tic()
            priorbox = PriorBox(cfg, image_size=(im_height, im_width))
            priors = priorbox.forward()
            priors = priors.to(device)
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
            boxes = boxes * scale / resize
            boxes = boxes.cpu().numpy()
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
            landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
            scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2]])
            scale1 = scale1.to(device)
            landms = landms * scale1 / resize
            landms = landms.cpu().numpy()

            # ignore low scores
            inds = np.where(scores > confidence_threshold)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]

            # keep top-K before NMS
            # order = scores.argsort()[::-SelfBlendedImages][:args.top_k]
            order = scores.argsort()[::-1]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]

            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, nms_threshold)

            dets = dets[keep, :]
            landms = landms[keep]

            # keep top-K faster NMS
            # dets = dets[:args.keep_top_k, :]
            # landms = landms[:args.keep_top_k, :]

            dets = np.concatenate((dets, landms), axis=1)
            _t['misc'].toc()

            ##

            bboxes, landmarks, faces = Mylign_multi(image, dets, vis_thres)
            bboxes = np.array(bboxes)
            if len(bboxes) == 0:
                # print('no face')
                continue
            else:
                bboxes = bboxes[:, :-1]  # shape:[10,4],only keep 10 highest possibiity faces
                bboxes = bboxes.astype(int)
                bboxes = bboxes + [-1, -1, 1, 1]  # personal choice
                conf = get_config(False)
                results, score = learner.infer(conf, faces, targets, True)

                for idx, bbox in enumerate(bboxes):
                    if args.score:
                        frame = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), frame)
                    else:
                        frame = draw_box_name(bbox, names[results[idx] + 1], frame)
            cv2.imshow('face Capture', frame)
            video_writer.write(frame)

        else:
            break
        if args.duration != 0:
            i += 1
            if i % 25 == 0:
                print('{} second'.format(i // 25))
            if i > 25 * args.duration:
                break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    video_writer.release()
    
