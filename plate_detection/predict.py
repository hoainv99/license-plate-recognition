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
    def __init__(self,detect_path='weights/best-checkpoint.bin',model_arc='weights/model_eff_arc.json', weights='weights/model_best_acc.h5'):
        #clf
        super().__init__()
        with open(model_arc, 'r') as f:
            self.model = tf.keras.models.model_from_json(f.read())
        self.model.load_weights(weights)
        self.mapper = [ 'TH', 'ACB', 'Acecook', 'Addidas', 'Agribank', 'Bidv', 'Big C', 'Cai Lan',
                        'Chinsu', 'Colgate', 'FPT', 'Habeco', 'Hai Ha', 'Jollibee', 'KFC', 'Kinh Do',
                        'Lotte mart', 'Mbbank new', 'Mbbank old', 'Neptune', 'Nike', 'Pepsi', 'Petrolimex','Phuc Long',
                        'Samsung', 'SHB', 'Techcombank', 'The Coffe House', 'The gioi di dong', 'TPbank',
                        'Vietcombank', 'Vietinbank', 'Viettel','Vinamilk', 'Vinfast', 'Vinmart', 'Vifon', 'Vnpt', 'Vpbank']
        #detect
        self.transform = A.Compose([
            A.Resize(height=512, width=512, p=1.0),
            ToTensor(),
        ])

        self.load_detect(detect_path)


    def preprocess_detect(self,path_img):
        image = cv2.imread(path_img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformer = self.transform(image=image)
        image = transformer['image']
        return image.unsqueeze(0).cpu().float()

    def preprocess_clf(self, batch):
        tensor = []
        for img in batch:
            img = cv2.resize(img, (224, 224))
            img = img / 255.0
            tensor.append(img)
        return np.array(tensor)

    def make_predictions(self,images, score_threshold=0.21):
        # images = torch.stack(images).cpu().float()
        predictions = []
        with torch.no_grad():
            det = self.net(images,{'img_scale': torch.tensor([1.]*images.shape[0]).float().cpu(), 'img_size': torch.tensor([512,512]).cpu()})
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
        config = get_efficientdet_config('tf_efficientdet_d5')
        config.image_size = [512,512]
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
        keep_idx = torchvision.ops.nms(torch.from_numpy(predictions[0][0]['boxes']),torch.from_numpy(predictions[0][0]['scores']),0.1)
        boxes = []
        scores = []
        labels = []
        for i in keep_idx:
            boxes.append(predictions[0][0]['boxes'][i])
            scores.append(predictions[0][0]['scores'][i])
            labels.append(predictions[0][0]['labels'][i])
        boxes = np.array(boxes).astype(np.float32).clip(min=0, max=511)
        image_original = cv2.imread(path_test)
        image_original = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
        h,w = image_original.shape[0],image_original.shape[1]
        batch_clf=[]
        boxes[:,0] = boxes[:,0]*(w/512)
        boxes[:,1] = boxes[:,1]*(h/512)
        boxes[:,2] = boxes[:,2]*(w/512)
        boxes[:,3] = boxes[:,3]*(h/512)
        for box in boxes:
            batch_clf.append(image_original[int(box[1]):int(box[3]),int(box[0]):int(box[2]),:])

        tensor = self.preprocess_clf(batch_clf)
        time_clf = time.time()
        print(tensor.shape)
        predictions = self.model.predict(tensor)
        print(f"clf:{time.time()-time_clf}")
        result = []
        for box,score, prediction in zip(boxes,scores, predictions):
            box[0] = int(box[0])
            box[1] = int(box[1])
            box[2] = int(box[2])
            box[3] = int(box[3])
            idx = int(np.argmax(prediction))
            result.append({'box':box, 'label':self.mapper[idx],'score_detect':score, 'score_clf':prediction[idx]})    
        return image_original,result
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_test", required=True, help="path to image test")
    args = parser.parse_args()
    predictor = Predictor()
    for img in os.listdir(args.folder_test):
        path = os.path.join(args.folder_test,img)
        image_original,results = predictor(path)
        src_img = Image.fromarray(image_original)
        img_pil = ImageDraw.Draw(src_img)
        for rs in results:
            shape = [(rs['box'][0],rs['box'][1]),(rs['box'][2],rs['box'][3])]
            img_pil.rectangle(shape,outline="black",width=2)
            # '-score_detect:'+str(rs['score_detect'])+
            img_pil.text((rs['box'][0]-10,rs['box'][1]-10),rs['label']+'-score_clf:'+str(rs['score_clf']),fill = "red")
        src_img.show(title="Nguyen Viet Hoai")

    