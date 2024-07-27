from helpers import get_settings
import pandas as pd
import numpy as np
import os
import ast
import cv2
import matplotlib.pyplot as plt
from PIL import ImageOps , Image
import torch
from tqdm import tqdm

settings = get_settings()

def resize_bboxes(data,max_width,max_height) :
    
    data['x'] =  (max_width / data['width'] ) * data['x']
    data['y'] = (max_height / data['height'] ) * data['y']
    data['w'] = (max_width / data['width'] ) * data['w'] 
    data['h'] = (max_height / data['height'] ) * data['h']

    return data


def get_images_and_bboxes(data) :
    images_and_bboxes = {key : value[['x','y','w','h','utf8_string']].values.tolist() for key, value in data.groupby('image_id')}
    return images_and_bboxes


def resize_image_and_padding(image,max_height,max_width) :
    resized_image = cv2.resize(image,(max_width,max_height))
    return resized_image




def preprocess_labels(annotations,bboxes_and_objects) :
    labels = []
    boxes = []
    
    for annotation in annotations :
        x , y , w , h , text = annotation
        x1 = x 
        y1 = y
        x2 = x + w
        y2 = y + h
        boxes.append([x1,y1,x2,y2])
        labels.append(1)

    bboxes_and_objects['box'] = torch.tensor(boxes, dtype=torch.int64)
    bboxes_and_objects['label'] = torch.tensor(labels, dtype=torch.int64)

    return bboxes_and_objects

def preprocess_image_for_text_recognizer(image_path,max_height,max_width,bbox,processor) :

    image = cv2.imread(image_path)
    x , y , w , h = bbox
    x , y , w , h = int(x) , int(y) , int(w) , int(h)
    image = image[y:y+h,x:x+w]
    image = resize_image_and_padding(image,max_height,max_width)
    
    pixel_values  = processor(image, return_tensors="pt").pixel_values.squeeze(0)
    return pixel_values

def preprocess_image_for_detector(image_path,max_height,max_width) :
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = resize_image_and_padding(image,max_height,max_width)
    image = image / 255
    image = torch.from_numpy(image)
    image = image.permute(2, 0, 1)
    return image

def flatten_boxes(df) :
    df[['x','y','w','h']] = df['bbox'].apply(lambda x : pd.Series(ast.literal_eval(x)))
    return df

def preprocess_text_recognizer_data(images_annotations,processor):
    data = images_annotations[['image_id', 'bbox', 'utf8_string']]
    images, targets = [], []
    max_target_length = 0
    
    for _, row in tqdm(data.iterrows()):
        text = str(row['utf8_string'])
        labels = processor.tokenizer(text, padding="max_length", max_length= 10 ).input_ids
        max_target_length = max(max_target_length,len(labels))
        images.append(row['image_id'])
        targets.append({"box": ast.literal_eval(row['bbox']), "text": text })

    return images, targets , max_target_length
            

def preprocess_detector_data(images_annotations,images_info_data,max_width , max_height) :

    data = pd.merge(images_annotations,images_info_data,left_on='image_id',right_on='id',how='left')
    data = flatten_boxes(data)
    data = data[['id_x','image_id','x','y','h','w','utf8_string','width','height']]
    data = resize_bboxes(data,max_width,max_height)
    data.to_csv(settings.PROCESSED_DETECTOR_DATA, encoding='utf-8' )

    images_and_bboxes = get_images_and_bboxes(data)
    images = []
    targets = []
    for image_id,bboxes in images_and_bboxes.items() :
        images.append(image_id)
        bboxes_and_objects = {}
        bboxes_and_objects = preprocess_labels(bboxes,bboxes_and_objects)
        targets.append(bboxes_and_objects)

    return images , targets