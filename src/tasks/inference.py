from .detector_model import BboxDetector
import torch
from helpers import get_settings
from .preprocess_data import resize_image_and_padding , preprocess_image_for_detector , preprocess_image_for_text_recognizer
from torchvision.ops import nms
import cv2
from .text_recognition_model import get_text_recognizer_model
import numpy as np
import matplotlib.pyplot as plt
from transformers import TrOCRProcessor

settings = get_settings()
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_classes = 2
detector_model = BboxDetector(num_classes=num_classes)
detector_model = detector_model.to(device)
detector_state_dict = torch.load(settings.SAVED_DETECTOR_MODEL_PATH)
detector_model.load_state_dict(detector_state_dict)
detector_model.eval()

text_recognizer_model = get_text_recognizer_model(processor=processor)
text_recognizer_model = text_recognizer_model.to(device)
recognizer_state_dict = torch.load(settings.SAVED_TEXT_RECOGNIZER_MODEL_PATH)
text_recognizer_model.load_state_dict(recognizer_state_dict)
max_height , max_width = 500 , 500

# Define arrow properties
arrow_color = (0, 255, 0)  # Green
arrow_thickness = 2

# Define text properties
text_color = (255, 0, 0)  # Red
font_scale = 0.5
font_thickness = 1

def display_image_and_bounding_box(image , bounding_boxes) :

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for bounding_box in bounding_boxes :
        x1 , y1 , x2 , y2 = map(int, bounding_box)
        image_rgb = cv2.rectangle(img=image_rgb,pt1=(x1,y1),pt2=(x2, y2) , color=(255,0,0),thickness = 4)

    return image_rgb

def get_predictions(image_path,nms_threshold,certainty_score) :
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    height_scale , width_scale = height/max_height , width/max_width
    image_copy = preprocess_image_for_detector(image_path=image_path,max_height=max_height,max_width=max_width)
    image_copy = [image_copy]

    image_copy[0] = image_copy[0].to(device).to(torch.float32)    
    output = detector_model(image_copy)

    boxes , scores = output[0]["boxes"] , output[0]["scores"]
    indices = nms(boxes, scores, nms_threshold)

    boxes , scores , indices = output[0]["boxes"].tolist() , output[0]["scores"].tolist() , indices.tolist()
    high_score_boxes = []

    for i in range(len(indices)) :
        index = indices[i]
        score = scores[index]
        box = boxes[index]
        
        if score >= certainty_score :
            x1,y1,x2,y2 = box 
            high_score_boxes.append([x1*width_scale,y1*height_scale,x2*width_scale,y2*height_scale])


    output_image = display_image_and_bounding_box(image,high_score_boxes)
    output_text = []

    # text recognition part
    for detected_text_box in high_score_boxes :
        x1,y1,x2,y2 = map(int, detected_text_box)
        cropped_image = image[y1:y2,x1:x2]
        resized_cropped_image = resize_image_and_padding(cropped_image,max_height,max_width)
        pixel_values  = processor(resized_cropped_image, return_tensors="pt").pixel_values
        generated_ids  = text_recognizer_model.generate(pixel_values.to(device))
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        output_text.append(generated_text)

        
    return output_image , output_text