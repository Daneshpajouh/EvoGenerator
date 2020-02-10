%reset
#====================================>>>>> Genetic Algorithm

import cv2
%pylab inline
import os
import numpy as np
import matplotlib.pyplot
import matplotlib.image as mpimg
from matplotlib import ticker
import itertools
import functools
import operator
import random
import heapq
import math
from keras.applications.densenet import DenseNet121
from keras.preprocessing import image
from keras.applications.densenet import preprocess_input, decode_predictions
#from keras.utils import multi_gpu_model
#import keras
from keras.models import load_model
import _pickle as cPickle
from matplotlib.pyplot import imsave as isave
from PIL import Image

target_img_idx = 954

#keras.backend.clear_session()

ROWS = 224
COLS = 224
CHANNELS = 3
"""
def change_model(model, new_input_shape=(None, ROWS, COLS, CHANNELS)):
    # replace input shape of first layer
    model._layers[0].batch_input_shape = new_input_shape

    # rebuild model architecture by exporting and importing via json
    new_model = keras.models.model_from_json(model.to_json())

    # copy weights from old model to new one
    for layer in new_model.layers:
        try:
            layer.set_weights(model.get_layer(name=layer.name).get_weights())
            print("Loaded layer {}".format(layer.name))
        except:
            print("Could not transfer weights for layer {}".format(layer.name))

    return new_model
"""

new_model = DenseNet121(weights='imagenet')

#new_model = change_model(model,new_input_shape=(None, ROWS, COLS, CHANNELS))
#new_model.save(("/content/drive/My Drive/AI/my_model_"+str(ROWS)+"_"+str(COLS)+".h5"))

new_model.summary()


#new_model = load_model("/content/drive/My Drive/AI/my_model_224_224.h5")
#new_model.summary()

#parallel_model = multi_gpu_model(model, gpus=2)
