import cv2
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications.xception import preprocess_input
from detect import *
import os, shutil
import json
import base64
from io import BytesIO
from PIL import Image
IMG_SIZE = 299
categories = ["ArtDecor","Hitech","Indochina","Industrial","Scandinavian" ]
save_crop_images_directory = r"./runs/detect"
model = tf.keras.models.load_model(r"./xception_model_2.h5")
import torch

def save_crop_images(image): 
    # Model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    # Inference
    results = model(image)
    crops = results.crop(save=True)

    predict_data = {} 
    predict_data=predict_crop_images(crops)
    return predict_data

def predict_crop_images(crops):
    predict_data = {}
    index = 0
    for img in crops:
                crop_img = img["im"]
                img_array = crop_img/255.0
                img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                new_array = np.expand_dims(img_array, axis=0)
                prediction = model.predict(new_array)
                #https://stackoverflow.com/q/43310681
                pil_img = Image.fromarray(crop_img)
                buff = BytesIO()
                pil_img.save(buff, format="JPEG")
                new_image_string = base64.b64encode(buff.getvalue()).decode("utf-8")
                #
                styles = {}
                for ratio,style in zip(prediction[0],categories):
                    styles.update({style: '{0:.10f}'.format(ratio)})
                predict_data.update({str(index): {"crop_img": new_image_string, "predict":styles }})
                index = index+1
    return predict_data



