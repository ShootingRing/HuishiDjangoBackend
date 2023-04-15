import os
from builtins import int, len, float, Exception
import cv2
import dlib
import torch
from PIL import Image
from torch import nn

from DeepFake_FaceRecognition.LiveAdv.AdvDetection.models.models import model_selection


def get_bounding_box(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


class DeepfakeDetect:

    def __init__(self):
        self.progress = 0
        self.save_file = ''
        self.model = None
        self.cuda = False
        self.transform = None
        self.face_detector = None
        self.FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
        self.THICKNESS = 2
        self.FONT_SCALE = 1

    def set_save_file(self, file_path, filename):
        if os.path.isdir(file_path):
            self.save_file = os.path.join(file_path)
        else:
            self.save_file = os.path.join(file_path)

    def detect(self, input_type, file_path, model_name='xception', save_file='', cuda=True):
        filename = file_path.split('/')[-1]

        self.set_save_file(save_file, save_file + filename)
        self.face_detector = dlib.get_frontal_face_detector()
        self.cuda = True if cuda and torch.cuda.is_available() else False
        self.model, self.transform = model_selection(model_name=model_name)
        if self.cuda:
            self.model.cuda()
        if input_type == 'image':
            image = cv2.imread(file_path)
            label, confidence, image = self.__detect_image(image)
            if label == '':
                return {'face_num': 0}
            else:
                cv2.imwrite(self.save_file, image)
                return {'label': label, 'confidence': confidence, 'face_num': 1, 'save_file': self.save_file}
        elif input_type == 'video':
            video = cv2.VideoCapture(file_path)
            return self.__detect_video(video)
        else:
            raise Exception('input type error')

    def __predict_with_model(self, image, post_function=nn.Softmax(dim=1)):
        """
        Predicts the label of an input image. Preprocesses the input image and
        casts it to cuda if required

        :param image: numpy image
        :param model: torch model with linear layer at the end
        :param post_function: e.g., softmax
        :param cuda: enables cuda, must be the same parameter as the model
        :return: prediction (1 = fake, 0 = real)
        """
        # Preprocess
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        preprocessed_image = self.transform(Image.fromarray(image))
        preprocessed_image = preprocessed_image.unsqueeze(0)
        if self.cuda:
            preprocessed_image = preprocessed_image.cuda()

        # Model prediction
        output = self.model(preprocessed_image)
        output = post_function(output)

        # Cast to desired
        confidence, prediction = torch.max(output, 1)  # argmax
        confidence, prediction = float(confidence.detach().cpu().numpy()), float(prediction.cpu().numpy())

        return int(prediction), confidence

    def __draw_box(self, image, label, confidence, face):
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y
        color = (0, 255, 0) if label == 'real' else (0, 0, 255)
        cv2.putText(image, '({}, {:.1f}%)'.format(label, confidence * 100), (x, y + h + 30), self.FONT_FACE,
                    self.FONT_SCALE, color, self.THICKNESS, 2)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        return image

    def __detect_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray, 1)
        face_num = 0
        label, confidence = '', 0.0
        if len(faces):
            # For now only take biggest face
            face = faces[0]
            height, width = image.shape[:2]
            x, y, size = get_bounding_box(face, width, height)
            cropped_face = image[y:y + size, x:x + size]

            # Actual prediction using our model
            prediction, confidence = self.__predict_with_model(cropped_face)
            # ------------------------------------------------------------------
            label = 'fake' if prediction == 1 else 'real'
            image = self.__draw_box(image, label, confidence, face)
            face_num += 1
        return label, confidence, image

    def __detect_video(self, video):
        face_num = 0
        fake_num = 0
        frames_num = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        confidences = 0.0
        codec = int(video.get(cv2.CAP_PROP_FOURCC))
        # fourcc = cv2.VideoWriter_fourcc(chr(codec & 0xFF), chr((codec >> 8) & 0xFF), chr((codec >> 16) & 0xFF),
        #                                 chr((codec >> 24) & 0xFF))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = None
        cnt = 0
        fakeconfi = 0.0
        realconfi = 0.0
        while video.isOpened():
            _, image = video.read()
            if image is None:
                break
            cnt += 1
            height, width = image.shape[:2]
            if video_writer is None:
                video_writer = cv2.VideoWriter(self.save_file,
                                               fourcc,
                                               video.get(cv2.CAP_PROP_FPS),
                                               (height, width)[::-1])

            if cnt % 5 == 0:
                label, confidence, image = self.__detect_image(image)
                if label != '':
                    face_num += 1
                    if label == 'fake':
                        fake_num += 1
                        fakeconfi += confidence
                    else:
                        realconfi += confidence
                self.progress = int(cnt / frames_num * 100)
            video_writer.write(image)
        self.progress = 100
        if video_writer is not None:
            video_writer.release()
        return {'fake_num': fake_num, 'face_num': face_num, 'avg_confidence': confidences / face_num,
                'save_file': self.save_file, 'ans': realconfi - fakeconfi}
