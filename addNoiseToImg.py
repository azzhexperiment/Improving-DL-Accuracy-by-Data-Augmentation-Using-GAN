import os
import sys
from skimage import io
from skimage import util
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, load_img


def get_data(mode, r, c, i):
    img_dir = "/home/students/student5_15/chest_xray/trial/pneumonia/"
    img_path = os.listdir(img_dir)
    out_dir = "/home/students/student5_15/chest_xray/trial/"

    print(img_path)

    for filename in img_path:
        img = skimage.io.imread(img_dir + filename)
        print(img_dir + filename)
#         img = img_to_array(load_img(img_dir + filename))
        plot_noise(img, mode, r, c, i)
        plt.savefig(out_dir + filename + '.jpeg')


def plot_noise(img, mode, r, c, i):
    plt.subplot(r, c, i)
    if mode is not None:
        gimg = skimage.util.random_noise(img, mode=mode)
        plt.imshow(gimg)
    else:
        plt.imshow(img)
    plt.title(mode)
    plt.axis("off")
    plt.show()


# Modifier
r = 4
c = 2
i = 1
mode = "gaussian"

get_data(mode, r, c, i)
