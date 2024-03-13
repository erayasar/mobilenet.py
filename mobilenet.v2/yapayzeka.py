import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

path= './data/'
Nudity=os.listdir(path+'nudity/')
Normal=os.listdir(path+'normal/')

vgg16 = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
preprocess_input = tf.keras.applications.vgg16.preprocess_input
image = tf.keras.preprocessing.image
batch_size=20

def extract_features(img_paths,batch_size=batch_size):
    global vgg16
    n = len(img_paths)
    img_array = np.zeros((n,224,224,3))

    for i, path in enumerate(img_paths):
        img = image.load_img(path, target_size=(224,224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        x = preprocess_input(img)
        img_array[i] = x

    X = vgg16.predict(img_array, batch_size=batch_size, verbose=1)
    X = X.reshape(n, 512, -1)

X = extract_features(
    list(map(lambda x: path + 'nudity/' + x, Nudity)) + list(map(lambda x: path +'normal/' + x, Normal))
)
y= np.array([1] *len(Nudity) + [0] * len(Normal))