from helpers import get_settings
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from tasks import getRecognizerLoader , train_text_recognizer , evaluate_text_recognizer , get_text_recognizer_model , preprocess_text_recognizer_data
import torch.nn as nn
from torch.optim import AdamW
from transformers import TrOCRProcessor

settings = get_settings()
images_annotations = pd.read_csv(settings.AANOTATIONS_PATH, encoding='utf-8')
images_info_data = pd.read_csv(settings.IMAGES_INFO_PATH, encoding='utf-8')
max_width , max_height = 500 , 500

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
images, boxes_and_labels , max_target_length =  preprocess_text_recognizer_data(images_annotations,processor)

training_images, testing_images, training_targets, testing_targets = train_test_split(images , boxes_and_labels , test_size= .2 , shuffle= True)
validation_images, testing_images, validation_targets, testing_targets = train_test_split(testing_images , testing_targets  , test_size= .5 , shuffle= True)

batch_size = 1
num_workers = 1
pin_memory = True
shuffle = True
epochs = 4

training_data_loader = getRecognizerLoader(images=training_images, targets=training_targets , num_workers = num_workers , shuffle=shuffle , pin_memory = pin_memory 
                                           ,max_height= max_height ,max_width= max_width, processor = processor , max_target_length = max_target_length ,batch_size = batch_size ).get_loader()
validation_data_loader = getRecognizerLoader(images=validation_images, targets=validation_targets  , num_workers = num_workers , shuffle=shuffle , pin_memory = pin_memory 
                                             ,max_height= max_height ,max_width= max_width, processor = processor , max_target_length = max_target_length,batch_size = batch_size ).get_loader()
testing_data_loader = getRecognizerLoader(images=testing_images, targets=testing_targets  , num_workers = num_workers , shuffle=True , pin_memory = pin_memory 
                                          ,max_height= max_height ,max_width= max_width, processor = processor , max_target_length = max_target_length,batch_size = batch_size ).get_loader()


text_recognizer_model = get_text_recognizer_model(processor=processor)
text_recognizer_model = text_recognizer_model.to(device)



optimizer = AdamW(text_recognizer_model.parameters(),lr = 0.00005)
best_score = float('inf')

for param_group in optimizer.param_groups:
    param_group['lr'] = 0.00005


for i in range(epochs) :
    training_loss = train_text_recognizer(model=text_recognizer_model , data_loader= training_data_loader , 
                                          optimizer=optimizer ,device=device )
    evaluation_loss , validation_cer = evaluate_text_recognizer(model=text_recognizer_model , data_loader= validation_data_loader , device=device , processor= processor)

    if evaluation_loss < best_score :
        best_score = evaluation_loss
        torch.save(text_recognizer_model.state_dict(),'text_recognizer_best_model.bin')

    print(f"Epoch {i} :\nTraining loss = {training_loss}\nEvaluation loss = {evaluation_loss}\nEvaluation CER = {validation_cer}")