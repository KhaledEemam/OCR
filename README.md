OCR Project
==============================

**Project Description**:

This OCR app is designed to detect and extract text from images. The project involves two main models:

* Text Detection: A Faster R-CNN model is trained to identify the regions in an image that contain text.
* Text Recognition: The identified text regions are then processed by a TrOCR model, which uses transformers to extract the text from these regions.

I developed an inference function that integrates these two models. First, the text detection model finds the text areas in the image. These areas are then passed to the text recognition model, which extracts the actual text.

Additionally, I created a FastAPI endpoint to allow users to interact with the models easily. To manage the input images, I used MongoDB to store them, organizing each image according to the project number. MongoDB was set up using a Docker image to simplify the installation process.

Technologies stack :
==============================
* Python
* Pytorch
* FastAPI
* MongoDB
* Motor
* Docker


Project Structure
==============================
```bash

├── README.md
├── docker
│   └── docker-compose.yml
└── src
    ├── Notebooks
    │   └── text-recognition- RNN-LSTM approaach.ipynb
    ├── assets
    │   ├── README_images
    │   │   ├── 1.png
    │   │   └── 2.png
    │   ├── data
    │   │   ├── preprocessed
    │   │   │   └── annot.csv
    │   │   └── raw
    │   │       ├── annot.csv
    │   │       └── img.csv
    │   ├── files
    │   ├── postman collections
    │   │   └── OCR.postman_collection.json
    │   └── trained_models
    │       ├── best_model.bin
    │       └── text_recognizer_best_model.bin
    ├── controllers
    │   ├── BaseController.py
    │   ├── DataController.py
    │   ├── ProjectController.py
    │   └── __init__.py
    ├── helpers
    │   ├── __init__.py
    │   └── config.py
    ├── main.py
    ├── models
    │   ├── AssetDataModel.py
    │   ├── BaseDataModel.py
    │   ├── ImagesDataModel.py
    │   ├── ProjectDataModel.py
    │   ├── __init__.py
    │   ├── db_schemas
    │   │   ├── Asset.py
    │   │   ├── Image.py
    │   │   ├── Project.py
    │   │   └── __init__.py
    │   └── enums
    │       ├── AssetEnum.py
    │       ├── DataBaseEnum.py
    │       ├── ResponseSignal.py
    │       └── __init__.py
    ├── requirements.txt
    ├── routes
    │   ├── __init__.py
    │   ├── base.py
    │   └── data.py
    ├── tasks
    │   ├── __init__.py
    │   ├── build_detector_data_loader.py
    │   ├── build_detector_dataset.py
    │   ├── build_recognizer_data_loader.py
    │   ├── build_recognizer_dataset.py
    │   ├── detector_model.py
    │   ├── evaluate_detector.py
    │   ├── evaluate_text_recognizer.py
    │   ├── inference.py
    │   ├── preprocess_data.py
    │   ├── text_recognition_model.py
    │   ├── train_detector.py
    │   └── train_text_recognizer.py
    ├── train_and_evaluate_detector.py
    └── train_and_evaluate_text_recognizer.py
```


Requirements
==============================
- Python 3.8 or later

## Install Python using MiniConda

1) Download and install MiniConda from [here](https://docs.anaconda.com/free/miniconda/#quick-command-line-install)
2) Create a new environment using the following command:
```bash
$ conda create -n mini-rag python=3.8
```
3) Activate the environment:
v
 Make sure that you have installed CUDA and cuDNN. Also make sure that you have Docker installed on your device if you are going to folloe the Docker installation instructions.

4) Download the dataset from 
https://www.kaggle.com/datasets/robikscube/textocr-text-extraction-from-images-dataset

4) Move train_images, 'annot.csv', and 'img.csv' into the following path :

```bash
$ \src\assets\data\raw
```

# **Installation**

## *Manual Configuration*

### Install Mongodb using the provided docker compose file

```bash
$ docker-compose up --build
```

### Install the required packages

```bash
$ pip install -r requirements.txt
```

### Setup the environment variables

```bash
$ cp .env.example .env
```

Set your environment variables in the `.env` file. Like USERNAME & PASSWORD in MONGODB_URL and make sure to do the same with the .env.example file that exists in docker folder.


### Train your text detection model

```bash
$ python train_and_evaluate_detector.py
```

### Train your text recognition model

```bash
$ python train_and_evaluate_text_recognizer.py
```

### Run the FastAPI server

```bash
$ uvicorn main:app --reload --port 5000
```

### Import the postman collection 
- open Postman
- Import the provided collection from "assets\postman collections\OCR.postman_collection.json"
- Navigate to the process API
- Import your image and give it a try.

### Output Sample

![](src\assets\README_images\1.png)
-------
![](src\assets\README_images\2.png)
--------