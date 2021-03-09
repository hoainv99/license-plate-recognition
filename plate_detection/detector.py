import gc
import tensorflow as tf
import cv2
import numpy as np
import torch
import os
import albumentations as A
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict 
from effdet.efficientdet import HeadNet
import torch.nn as nn
import  matplotlib.pyplot as plt
from albumentations.pytorch.transforms import ToTensor
import torchvision
import argparse
from PIL import Image, ImageDraw, ImageFont
import time
class Predictor(nn.Module):
    def __init__(self,detect_path):
        super().__init__()
        #detect
        self.transform = A.Compose([
            A.Resize(height=384, width=384, p=1.0),
            ToTensor(),
        ])
        self.load_detect(detect_path)
    def preprocess_detect(self,path_img):
        image = cv2.imread(path_img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformer = self.transform(image=image)
        image = transformer['image']
        return image.unsqueeze(0).cpu().float()
    def make_predictions(self,images, score_threshold=0.25):
        # images = torch.stack(images).cpu().float()
        predictions = []
        with torch.no_grad():
            det = self.net(images,{'img_scale': torch.tensor([1.]*images.shape[0]).float().cpu(), 'img_size': torch.tensor([384,384]).cpu()})
            for i in range(images.shape[0]):
                boxes = det[i].detach().cpu().numpy()[:,:4]    
                scores = det[i].detach().cpu().numpy()[:,4]
                labels = det[i].detach().cpu().numpy()[:,5]
                indexes = np.where(scores > score_threshold)[0]
                predictions.append({
                    'boxes': boxes[indexes],
                    'scores': scores[indexes],
                    'labels':labels[indexes]
                })
        return [predictions]
    def load_detect(self,checkpoint_path):
        config = get_efficientdet_config('tf_efficientdet_d0')
        config.image_size = [384,384]
        net = EfficientDet(config, pretrained_backbone=False)
        net.reset_head(num_classes=1)
        checkpoint = torch.load(checkpoint_path,map_location=torch.device('cpu'))
        net.load_state_dict(checkpoint['model_state_dict'])
        del checkpoint
        gc.collect()
        self.net = DetBenchPredict(net)
        self.net.eval()
        self.net.cpu()
    def forward(self,path_test):
        images = self.preprocess_detect(path_test)
        s_t = time.time()
        predictions = self.make_predictions(images)
        print(time.time()-s_t)
        batch_recognize=[]
        boxes = []
        scores = []
        labels = []
        if(predictions[0][0]['boxes']!=[]):
            keep_idx = torchvision.ops.nms(torch.from_numpy(predictions[0][0]['boxes']),torch.from_numpy(predictions[0][0]['scores']),0.1)
            for i in keep_idx:
                boxes.append(predictions[0][0]['boxes'][i])
                scores.append(predictions[0][0]['scores'][i])
                labels.append(predictions[0][0]['labels'][i])
            boxes = np.array(boxes).astype(np.float32).clip(min=0, max=511)
            image_original = cv2.imread(path_test)
            image_original = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
            h,w = image_original.shape[0],image_original.shape[1]
            boxes[:,0] = boxes[:,0]*(w/384)
            boxes[:,1] = boxes[:,1]*(h/384)
            boxes[:,2] = boxes[:,2]*(w/384)
            boxes[:,3] = boxes[:,3]*(h/384)
            for box in boxes:
                batch_recognize.append(image_original[int(box[1]):int(box[3]),int(box[0]):int(box[2]),:])  
        return batch_recognize,boxes


    
