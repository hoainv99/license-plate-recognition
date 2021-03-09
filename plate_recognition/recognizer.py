import torch.nn as nn
from .modules.utils import build_model, translate
import torch
from torchvision import transforms
import math
import copy
from PIL import Image
import numpy as np
import cv2


class NormalizePAD(object):

    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w -
                                    1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img

class Recognizer(nn.Module):
    def __init__(self, path_model):
        super().__init__()
        vocab = '0J2O3Q5BVSG9XT7IPYFWZEURHA8NMDL1K6C4'
        self.model, self.vocab = build_model(vocab)

        self.device = torch.device('cpu')

        self.load_weights(path_model)
        self.imgH = 128
        self.imgW = 384
        resized_max_w = self.imgW
        input_channel = 3
        self.transform = NormalizePAD(
            (input_channel, self.imgH, resized_max_w))

    def load_weights(self, filename):
        state_dict = torch.load(
            filename, map_location=torch.device(self.device))

        for name, param in self.model.named_parameters():
            if name not in state_dict:
                print('{} not found'.format(name))
            elif state_dict[name].shape != param.shape:
                print('{} missmatching shape, required {} but found {}'.format(
                    name, param.shape, state_dict[name].shape))
                del state_dict[name]

        self.model.load_state_dict(state_dict, strict=False)

    def process_input(self, image):

        w, h = image.size
        ratio = w / float(h)
        if math.ceil(self.imgH * ratio) > self.imgW:
            resized_w = self.imgW
        else:
            resized_w = math.ceil(self.imgH * ratio)

        resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)

        return self.transform(resized_image)

    def forward(self, list_img):
        resized_images = []
        for img in list_img:
            img = Image.fromarray(img).convert('RGB')
            img = self.process_input(img)
            resized_images.append(img)
        image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)
        image_tensors = image_tensors.to(self.device)
        translated_sentence = translate(image_tensors, self.model)
        pred_sents = self.vocab.batch_decode(translated_sentence.tolist())
        return pred_sents