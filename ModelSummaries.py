from tensorflow.keras import models

cats_v_dogs = models.load_model('cats-vs-dogs-model.h5')

cifar = models.load_model('cifar-10-model.h5')

minst = models.load_model('minst-model.h5')

fashion = models.load_model('minst-fashion-model.h5')

print(cats_v_dogs.summary())
print(cifar.summary())
print(minst.summary())
print(fashion.summary())


