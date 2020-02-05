from keras.utils.np_utils import to_categorical  # convert to one-hot-encoding
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.utils.np_utils import to_categorical  # convert to one-hot-encoding
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as pimg
import os
import matplotlib

print(os.listdir("chest-xray-pneumonia/chest_xray/chest_xray"))

main_path = "chest-xray-pneumonia/chest_xray/chest_xray"
train_path = main_path + "/train/"
val_path = main_path + "/val/"
test_path = main_path + "/test/"

train_n = train_path + "NORMAL"
train_p = train_path + "PNEUMONIA"
val_n = val_path + "NORMAL"
val_p = val_path + "PNEUMONIA"
test_n = test_path + "NORMAL"
test_p = test_path + "PNEUMONIA"

# get the data, label it and normalize it
train_batches = ImageDataGenerator(rescale=1./255).flow_from_directory(
    train_path, target_size=(64, 64), classes=["NORMAL", "PNEUMONIA"], batch_size=10)
val_batches = ImageDataGenerator(rescale=1./255).flow_from_directory(
    val_path, target_size=(64, 64), classes=["NORMAL", "PNEUMONIA"], batch_size=4)
test_batches = ImageDataGenerator(rescale=1./255).flow_from_directory(
    test_path, target_size=(64, 64), classes=["NORMAL", "PNEUMONIA"], batch_size=10)


len(train_batches.labels)
len(test_batches.labels)
len(val_batches.labels)

# taken from https://github.com/smileservices/keras_utils/blob/master/utils.py
# plot images with labels


def plots(ims, figsize=(12, 6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0, 2, 3, 1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')


show_images = ImageDataGenerator().flow_from_directory(
    train_path, target_size=(64, 64), classes=["NORMAL", "PNEUMONIA"], batch_size=10)
imgs, labels = show_images[0]
plots(imgs, titles=labels)


model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)),
    MaxPool2D(pool_size=(3, 3)),
    Conv2D(32, (3, 3), activation="relu"),
    MaxPool2D(pool_size=(2, 2)),
    Flatten(),
    Dense(16, activation="relu"),
    Dense(2, activation="relu")
])
model.compile(optimizer="sgd", loss="mean_squared_error", metrics=["accuracy"])


model.summary()


con_network = model.fit_generator(train_batches, steps_per_epoch=(
    5216/10), epochs=5, validation_data=val_batches, validation_steps=100, verbose=2)

accuracy_test = model.evaluate_generator(test_batches, steps=624)

print(f"The test accuracy is: {accuracy_test[1]*100}")

plt.plot(con_network.history["accuracy"])
plt.plot(con_network.history["val_accuracy"])
plt.title("Accuracy of the model")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Training set", "Validation set"], loc="upper left")
plt.show()


plt.plot(con_network.history["loss"])
plt.plot(con_network.history["val_loss"])
plt.title("Accuracy of the model")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Training set", "Validation set"], loc="upper left")
plt.show()
