# import tensorflow as tf
# from tensorflow import keras
import json
import cv2 as cv

def save_model_to_json(model, path="model.json"):
    json_config = model.to_json()
    d = json.loads(json_config)
    with open(path, 'w') as fp:
        json.dump(d,fp,indent=4)
    print("Model saved to " + path)

def augment_img(img):
    
    return