import numpy as np # linear algebra
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D, Dropout
import datetime
import os

print(os.listdir('./'))

train_labels = []
train_images = []
test_images = []
test_labels = []

def get_train_data():
    for file in os.listdir('dogs-vs-cats/train'):
        category = file.split('.')[0]
        if category == 'dog':
            category = 1
        else:
            category = 0

        img_array = cv2.imread(os.path.join('dogs-vs-cats/train', file), cv2.IMREAD_GRAYSCALE)
        if img_array is not None:
            new_img_array = cv2.resize(img_array, dsize=(80, 80))
            train_images.append(new_img_array)
            train_labels.append(category)


def get_test_data():
    for file in os.listdir('dogs-vs-cats/test'):
        category = file.split('.')[0]
        if category == 'dog':
            category = 1
        else:
            category = 0
        img_array = cv2.imread(os.path.join('dogs-vs-cats/test', file), cv2.IMREAD_GRAYSCALE)
        if img_array is not None:
            new_img_array = cv2.resize(img_array, dsize=(80, 80))
            test_images.append(new_img_array)
            test_labels.append(category)


get_train_data()
get_test_data()


train_images = np.array(train_images).reshape(-1, 80, 80, 1)
test_images = np.array(test_images).reshape(-1, 80, 80, 1)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

train_images = train_images / 255.0
test_images = test_images / 255.0

model = Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(Conv2D(64,(3,3), activation = 'relu', input_shape = train_images.shape[1:]))
model.add(MaxPooling2D(pool_size = (2,2)))
# Add another:
model.add(Conv2D(64,(3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer="adam",
              loss='binary_crossentropy',
              metrics=['accuracy'])

start_train_time = datetime.datetime.now()
model.fit(train_images, train_labels, epochs=10, validation_split=0.2)
stop_train_time = datetime.datetime.now()

model.save('cats-vs-dogs-model.h5')

start_test_time = datetime.datetime.now()
test_loss, test_acc = model.evaluate(test_images, test_labels)
stop_test_time = datetime.datetime.now()

print(test_acc)
print("Train Time: " + str((stop_train_time - start_train_time).total_seconds()))
print("Test Time: " + str((stop_test_time - start_test_time).total_seconds()))



