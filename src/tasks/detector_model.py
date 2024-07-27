import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class BboxDetector(nn.Module) :
    def __init__(self,num_classes) :
        super(BboxDetector,self).__init__()
        self.detector = models.detection.fasterrcnn_resnet50_fpn()
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes)

    def forward(self,images,targets=None) :
        if targets == None :
            output = self.detector(images)
        else :
            output = self.detector(images,targets)
            
        return output