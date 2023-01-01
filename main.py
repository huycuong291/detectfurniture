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

def save_crop_images(image):
    folder = save_crop_images_directory
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    cv2.imwrite('./data/images/client_side_image.jpg', image)
    run( **{'weights': 'yolov5s.pt', 'source': './data/images/client_side_image.jpg', 'data': './data/coco128.yaml', 'conf_thres': 0.35, 'iou_thres': 0.45, 'max_det': 1000, 'device': '', 'view_img': False, 'save_txt': False, 'save_conf': False, 'save_crop': True, 'nosave': True, 'classes': None, 'agnostic_nms': False, 'augment': False, 'visualize': False, 'update': False, 'project': './runs/detect', 'name': 'exp', 'exist_ok': False, 'line_thickness': 3, 'hide_labels': False, 'hide_conf': False, 'half': False, 'dnn': False, 'vid_stride': 1})

def predict_crop_images():
    predict_data = {}
    index = 0
    for path in os.listdir(save_crop_images_directory):
        path2 = os.path.join(save_crop_images_directory, path)
        if os.listdir(path2)== []: return "ERROR"
        for new_path in os.listdir(path2 + '\crops'):
            path3 = os.path.join(path2 + '\crops', new_path)
            for img in os.listdir(path3):
                crop_img = cv2.imread(os.path.join(path3, img), cv2.IMREAD_COLOR)
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
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



