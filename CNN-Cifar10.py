from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
import datetime
from tensorflow.keras import datasets, layers, models

def rgb2gray(rgb_images):
    images = []
    for rgb in rgb_images:
        images.append(np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140]))
    return np.array(images)

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images = train_images.reshape((50000, 32, 32, 3))
test_images = test_images.reshape((10000, 32, 32, 3))

train_images = rgb2gray(train_images)
test_images = rgb2gray(test_images)

train_images = train_images.reshape((50000, 32, 32, 1))
test_images = test_images.reshape((10000, 32, 32, 1))

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

start_train_time = datetime.datetime.now()
model.fit(train_images, train_labels, epochs=10, validation_split=0.2)
stop_train_time = datetime.datetime.now()

model.save('cifar-10-model.h5')

start_test_time = datetime.datetime.now()
test_loss, test_acc = model.evaluate(test_images, test_labels)
stop_test_time = datetime.datetime.now()

print(test_acc)
print("Train Time: " + str((stop_train_time - start_train_time).total_seconds()))
print("Test Time: " + str((stop_test_time - start_test_time).total_seconds()))

