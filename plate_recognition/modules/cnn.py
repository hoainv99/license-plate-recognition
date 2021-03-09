import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import math
class effnet(nn.Module):
    def __init__(self, d_model,name = 'efficientnet-b4',dropout=0.5):
        super().__init__()
        md = EfficientNet.from_pretrained(name)
        self.model = nn.Sequential(*list(md.children())[0:2],
             nn.Sequential(*list(md.children())[2]),
             *list(md.children())[3:5]) 
        self.dropout = nn.Dropout(dropout)
        self.last_conv_1x1 = nn.Conv2d(1792, d_model, 1)
    def forward(self,x):
        return  self.last_conv_1x1(self.dropout(self.model(x)))
    
    
class CNN(nn.Module):
    def __init__(self, d_model=256):
        super(CNN, self).__init__()
        self.model = effnet(d_model)   
        self.d_model = d_model

    def forward(self, x):
        return self.model(x)*math.sqrt(self.d_model)