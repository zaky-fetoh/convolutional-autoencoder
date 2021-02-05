import keras.datasets as dts
import keras.preprocessing.image as image
import keras.utils as utils
import numpy as np


batch_size = 64

def get_data():
    return dts.mnist.load_data()


def get_gen():
    (tr_im, tr_lb), (ts_im, ts_lb) = get_data()
    tr_gen = image.ImageDataGenerator(rotation_range=45,
                                      width_shift_range=.2,
                                      height_shift_range=.2,
                                      shear_range=.2,
                                      zoom_range=.2,
                                      horizontal_flip=True,
                                      vertical_flip=True,
                                      rescale=1. / 255)
    ts_gen = image.ImageDataGenerator(rescale=1. / 255)

    tr_lb, ts_lb = utils.to_categorical(tr_lb), utils.to_categorical(ts_lb)

    tr_gen = tr_gen.flow(tr_im.reshape(tr_im.shape + (1,)), tr_lb, batch_size=batch_size)
    ts_gen = ts_gen.flow(ts_im.reshape(ts_im.shape + (1,)), ts_lb, batch_size=batch_size)

    return tr_gen, ts_gen


def im_gen(tr_gen):
    for ims, lbs in tr_gen :
        yield (ims, ims )


def get_generators():
    tr_gen, val_gen = get_gen();
    return im_gen(tr_gen), im_gen(val_gen)



