from re import I
from models import exif
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import Input
from keras import Model,Sequential
from keras.initializers import RandomNormal
from keras.layers import Dense,Flatten,Dropout,Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from extract_exif import extract_exif, random_list,generate_label,cropping_list,get_np_arrays
from matplotlib import image
from lib.utils import benchmark_utils, util,io
import cv2
import numpy as np
import keras
import pickle
from keras.engine import keras_tensor


EPOCHS = 100


list1,list2 = get_np_arrays('cropped_arrays.npy')


with open("exif_lbl.txt", "rb") as fp:   #Picklingpickle.dump(l, fp)
	exif_lbl = pickle.load(fp)
fp.close()

def datagenerator(images,images2, labels, batchsize, mode="train"):
    while True:
        start = 0
        end = batchsize
        while start  < len(images):
            #if(len(images)-start < batchsize):
            #    break
            # load your images from numpy arrays or read from directory
            #else:
            x = images[start:end] 
            y = labels[start:end]
            x2 = images2[start:end]
            yield (x,x2),y

            start += batchsize
            end += batchsize

def create_base_model(image_shape, dropout_rate, suffix=''):
    I1 = Input(image_shape)
    model = ResNet50(include_top=False, weights='imagenet', input_tensor=I1, pooling=None)
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1]._outbound_nodes = []

    for layer in model.layers:
        layer._name = layer.name + str(suffix)
        layer._trainable = False

    flatten_name = 'flatten' + str(suffix)

    x = model.output
    x = Dense(256, activation='relu')(x)
    x = Flatten(name=flatten_name)(x)
    

    return x, model.input


def create_siamese_model(image_shape, dropout_rate):

    
    output_left, input_left = create_base_model(image_shape, dropout_rate)
    output_right, input_right = create_base_model(image_shape, dropout_rate, suffix="_2")
    
    output_siamese = tf.concat([output_left,output_right],1)
    num_classes=45;
    
    y = output_siamese
    y= Dense(4096, activation='relu')(y)
    y = Dense(2048, activation='relu')(y)
    y = Dense(1024, activation='relu')(y)
    y = Dense(num_classes, activation='softmax')(y)
    
    return y,input_left,input_right
    
    
def create_mlp(image_shape,dropout_rate):
    out,input_left,input_right = create_siamese_model(image_shape,
                                      dropout_rate)
                                      
    sm_model = Model(inputs=[input_left, input_right], outputs=out)
    
    return sm_model
    

total_model=create_mlp(image_shape=(128,128,3),dropout_rate=0.2)

total_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


with open("exif_lbl.txt", "rb") as fp:   #Picklingpickle.dump(l, fp)
	exif_lbl = pickle.load(fp)
fp.close()

for i in range(len(exif_lbl)):
    exif_lbl[i] = np.array(exif_lbl[i])
exif_lbl = np.array(exif_lbl)

#######################################################################################à
#crop images to 128x128
#######################################################################################à
list1,list2 = get_np_arrays('cropped_arrays.npy')
x_train = datagenerator(list1,list2,exif_lbl,16)

steps = 80

total_model.fit(x_train,epochs=EPOCHS,steps_per_epoch=steps)

