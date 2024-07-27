from torch.utils.data import Dataset
from torchvision import transforms
import cv2
from helpers import get_settings
import os
from .preprocess_data import preprocess_image_for_text_recognizer
import torch

settings = get_settings()

class CustomRecognizerDataset(Dataset) :
    def __init__(self,images,targets,max_height,max_width,processor,max_target_length) :
        self.max_height = max_height
        self.max_width = max_width
        self.image_id , self.targets = images , targets
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self) :
        return len(self.image_id)
    
    def __getitem__(self,index) :
        image_path = os.path.join(settings.IMAGES_PATH,self.image_id[index]+'.jpg')  
        pixel_values = preprocess_image_for_text_recognizer(image_path,self.max_height,self.max_width,bbox=self.targets[index]['box'],processor=self.processor)
         # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(self.targets[index]['text'], 
                                          padding="max_length", 
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        
        return {"pixel_values" : pixel_values , "labels" : torch.tensor(labels)}