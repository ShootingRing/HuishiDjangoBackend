import os
from builtins import Exception

import torch
from torchvision import transforms, models

from DeepFake_FaceRecognition.LiveAdv.AdvDetection.models.xception import Xception

pretrained = {
    'xception': os.path.abspath('G:\DeepFake_FaceRecognition (3)\DeepFake_FaceRecognition\LiveAdv/AdvDetection/weights/xception.pt'),
}


def model_selection(model_name, cuda=False):
    # adv检测模型
    ml = 'cpu'
    if cuda:
        ml = 'cuda'
    if model_name == 'xception':
        model = Xception(2)
        model.load_state_dict(torch.load(pretrained['xception'], map_location=ml))
        model.eval()
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])
    else:
        raise Exception('Invalid Model Name')
    return model, transform
