from PIL import Image
import json
import pandas as pd
import numpy as np
import os
from helpers import get_settings
from tasks import BboxDetector
import torch
from tasks import preprocess_detector_data , getDetectorLoader
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tasks import train_detector_epoch , eval_detector_epoch


settings = get_settings()
images_annotations = pd.read_csv(settings.AANOTATIONS_PATH)
images_info_data = pd.read_csv(settings.IMAGES_INFO_PATH)
max_width , max_height = 1024 , 1024

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

images , targets = preprocess_detector_data(images_annotations,images_info_data,max_width=max_width,max_height=max_height)

training_images, testing_images, training_targets, testing_targets = train_test_split(images , targets , test_size= .2 , shuffle= True)
validation_images, testing_images, validation_targets, testing_targets = train_test_split(testing_images , testing_targets  , test_size= .5 , shuffle= True)

batch_size = 1
num_workers = 1
pin_memory = True
shuffle = True
epochs = 10


training_data_loader = getDetectorLoader(images=training_images, targets=training_targets,batch_size = batch_size , num_workers = num_workers 
                                 , shuffle=shuffle , pin_memory = pin_memory ,max_height= max_height ,max_width= max_width ).get_loader()
validation_data_loader = getDetectorLoader(images=validation_images, targets=validation_targets ,batch_size = batch_size , num_workers = num_workers 
                                   , shuffle=shuffle , pin_memory = pin_memory ,max_height= max_height ,max_width= max_width).get_loader()
testing_data_loader = getDetectorLoader(images=testing_images, targets=testing_targets , batch_size = batch_size , num_workers = num_workers 
                                , shuffle=True , pin_memory = pin_memory ,max_height= max_height ,max_width= max_width).get_loader()

num_classes = 2
detector_model = BboxDetector(num_classes=num_classes)
detector_model = detector_model.to(device)

state_dict = torch.load(settings.SAVED_DETECTOR_MODEL_PATH)
detector_model.load_state_dict(state_dict)

optimizer = AdamW(detector_model.parameters(),lr = .0001)
training_steps = epochs * len(training_data_loader)
warmup_steps = int(.2 * training_steps)
scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,num_warmup_steps = warmup_steps , num_training_steps = training_steps)

best_score = 0

for i in range(epochs) :
    training_loss = train_detector_epoch(model=detector_model,optimizer=optimizer,scheduler=scheduler,
                                         data_loader=training_data_loader,device=device)
    validation_loss = eval_detector_epoch(model=detector_model,data_loader=validation_data_loader,device=device)

    if validation_loss > best_score :
        best_score = validation_loss
        torch.save(detector_model.state_dict(),settings.SAVED_DETECTOR_MODEL_PATH)

    print(f"Epoch {i} :\nTraining loss = {training_loss}\nEvaluation loss = {validation_loss}")