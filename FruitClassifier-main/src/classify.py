import os
import numpy as np
import tensorflow as tf
import cv2
import config
from preprocessing import DataLoader
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def preprocess_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f'image not found at {image_path}')
    img=image.load_img(image_path,target_size=(224,224))
    img_array=image.img_to_array(img)
    img_array=np.expand_dims(img_array,axis=0)
    img_array=preprocess_input(img_array)
    return img_array
def predict(image_path):
    loaded_model=tf.keras.models.load_model(config.MODEL_SAVE_PATH)
    imag=preprocess_image(image_path)
    predictions=loaded_model.predict(imag)
    pred_class=np.argmax(predictions)
    classes=['Apple', 'Banana', 'avocado', 'cherry', 'kiwi', 'mango', 'orange', 'pinenapple', 'strawberries', 'watermelon']
    print('prediction is : ',classes[pred_class])
def show_img(image_path):
    test=cv2.imread(image_path,-1)
    cv2.imshow('selected image',test)
    cv2.waitKey(0)
    
if __name__=="__main__":
    demo_img="./data/MY_data/predict/99.jpeg"
    predict(demo_img)
    show_img(demo_img)