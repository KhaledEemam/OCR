from torch.utils.data import Dataset
import cv2
from helpers import get_settings
import os
from .preprocess_data import preprocess_image_for_detector
import torch

settings = get_settings()

class CustomDetectorDataset(Dataset) :
    def __init__(self,images,targets,max_height,max_width) :
        self.max_height = max_height
        self.max_width = max_width
        self.image_id , self.encodings = images , targets

    def __len__(self) :
        return len(self.image_id)
    
    def __getitem__(self,index) :
        image_path = os.path.join(settings.IMAGES_PATH,self.image_id[index]+'.jpg')
        image = preprocess_image_for_detector(image_path,self.max_height,self.max_width)
        encoding = self.encodings[index]

        return {"images" : image , "targets" : encoding}