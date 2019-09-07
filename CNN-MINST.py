from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import datasets, layers, models
import datetime


(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

plt.imshow(train_images.item(0))
plt.show()
exit()


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

print(model.summary())

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

start_train_time = datetime.datetime.now()
model.fit(train_images, train_labels, epochs=5, validation_split=0.2)
stop_train_time = datetime.datetime.now()

model.save('minst-model.h5')

start_test_time = datetime.datetime.now()
test_loss, test_acc = model.evaluate(test_images, test_labels)
stop_test_time = datetime.datetime.now()

print(test_acc)
print("Train Time: " + str((stop_train_time - start_train_time).total_seconds()))
print("Test Time: " + str((stop_test_time - start_test_time).total_seconds()))
