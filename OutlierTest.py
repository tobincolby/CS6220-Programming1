from tensorflow.keras import models
import cv2
import os
import numpy as np
from datetime import datetime

cats_v_dogs = models.load_model('cats-vs-dogs-model.h5')

cifar = models.load_model('cifar-10-model.h5')

minst = models.load_model('minst-model.h5')

fashion = models.load_model('minst-fashion-model.h5')

class_names_fashion = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
class_names_handwriting = ['0', '1', '2', '3', '4' , '5', '6', '7', '8', '9']
class_names_cats_dogs = ['cat', 'dog']
class_names_cifar = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
test_images = []

def get_test_data_cats_classifier():
    for file in os.listdir('outliers'):
        img_array = cv2.imread(os.path.join('outliers', file), cv2.IMREAD_GRAYSCALE)
        if img_array is not None:
            new_img_array = cv2.resize(img_array, dsize=(80, 80))
            test_images.append(new_img_array)

def get_test_data_minst_classifier():
    for file in os.listdir('outliers'):
        category = file.split('.')[0]
        img_array = cv2.imread(os.path.join('outliers', file), cv2.IMREAD_GRAYSCALE)
        if img_array is not None:
            new_img_array = cv2.resize(img_array, dsize=(28, 28))
            test_images.append(new_img_array)

def get_test_data_cifar_classifier():
    for file in os.listdir('outliers'):
        img_array = cv2.imread(os.path.join('outliers', file), cv2.IMREAD_GRAYSCALE)
        if img_array is not None:
            new_img_array = cv2.resize(img_array, dsize=(32, 32))
            test_images.append(new_img_array)

# For MNIST Data Classifiers

get_test_data_minst_classifier()
test_images = np.array(test_images).reshape(-1, 28, 28, 1)
test_images = test_images / 255.0

start_time = datetime.now()
minst_predictions = minst.predict(test_images)
stop_time = datetime.now()

print("Handwriting")
print("TestTime: " + str((stop_time - start_time).total_seconds()))
for prediction in minst_predictions:
    print(class_names_handwriting[prediction.argmax()])

start_time = datetime.now()
fashion_predictions = minst.predict(test_images)
stop_time = datetime.now()
print("Fashion")
print("TestTime: " + str((stop_time - start_time).total_seconds()))
for prediction in fashion_predictions:
    print(class_names_fashion[prediction.argmax()])




#For Cifar Data Classifier
test_images = []
get_test_data_cifar_classifier()
test_images = np.array(test_images).reshape(-1, 32, 32, 1)
test_images = test_images / 255.0

start_time = datetime.now()
cifar_predictions = cifar.predict(test_images)
stop_time = datetime.now()
print("Cifar")
print("TestTime: " + str((stop_time - start_time).total_seconds()))
for prediction in cifar_predictions:
    print(class_names_cifar[prediction.argmax()])

#For Cats v Dogs Classifier
test_images = []
get_test_data_cats_classifier()
test_images = np.array(test_images).reshape(-1, 80, 80, 1)
test_images = test_images / 255.0

start_time = datetime.now()
cats_v_dogs_predictions = cats_v_dogs.predict(test_images)
stop_time = datetime.now()
print("Cats v Dogs")
print("TestTime: " + str((stop_time - start_time).total_seconds()))
for prediction in cats_v_dogs_predictions:
    print(class_names_cats_dogs[prediction.argmax()])






