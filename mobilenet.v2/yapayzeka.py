import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout,Dense
np.random.seed(42)
epochs = 5

path= './data/'
test_path= './data/test/'
Nudity = os.listdir(path + 'nudity/')
Normal = os.listdir(path + 'normal/')
Test = os.listdir(test_path)

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
    return X

X = extract_features(
    list(map(lambda x: path + 'nudity/' + x, Nudity)) + list(map(lambda x: path +'normal/' + x, Normal))
)
y= np.array([1] *len(Nudity) + [0] * len(Normal))

X_test = extract_features(
    list(map(lambda x: test_path + x, Test))
)
y_test = np.array([1] * len(Nudity) + [0] * len(Normal)) 

def train():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1724, activation=tf.nn.relu),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

    ])
    return model

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = train()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
history = model.fit(X_train,y_train,
                    validation_data=(X_test,y_test),
                    batch_size=batch_size,
                    epochs=epochs)

plt.plot(range(1,epochs+1), history.history['accuracy'], label='Train Accuracy')
plt.plot(range(1,epochs+1), history.history['val_accuracy'], label='Test Accuracy')
plt.plot(range(1,epochs+1), history.history['loss'], label='Train Loss')
plt.plot(range(1,epochs+1), history.history['val_loss'], label='Test Loss')
plt.legend()
plt.show()

X_test = extract_features(
    list(map(lambda x: './data/test/4463998783.jpg' , Test))
)
y_pred = model.predict(X_test)
if y_pred.all() > 0.5:
    print("Bu fotograf çıplaklık içermektedir.")
    print("Tahmin değeri:", y_pred)
else:
    print("Bu fotograf güvenlidir.")
    print("Tahmin değeri:", y_pred)
