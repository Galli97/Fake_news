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


EPOCHS = 10
steps=40
image_shape=(128,128,3)
dropout_rate=0.2
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
x_train = datagenerator(list1,list2,exif_lbl,32)

#steps = len(list1)/EPOCHS

with open("exif_lbl.txt", "rb") as fp:   #Picklingpickle.dump(l, fp)
	exif_lbl = pickle.load(fp)
fp.close()


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
    x = Dense(256, activation='softmax')(x)
    x = Flatten(name=flatten_name)(x)
    print(x.shape)

    return x, model.input


def create_siamese_model(image_shape, dropout_rate):

    
    output_left, input_left = create_base_model(image_shape, dropout_rate)
    output_right, input_right = create_base_model(image_shape, dropout_rate, suffix="_2")
    
    output_siamese = tf.concat([output_left,output_right],1)
    L1_layer = Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([output_left, output_right])
    L1_prediction = Dense(1, use_bias=True,
                          activation='sigmoid',
                          input_shape = image_shape,
                          kernel_initializer=RandomNormal(mean=0.0, stddev=0.001),
                          name='weighted-average')(L1_distance)

    prediction = Dropout(0.2)(L1_prediction)
    
    
    #model.summary()
    #siamese_model = Model(inputs=[input_left, input_right], outputs=output_siamese)
    #out = model.output
    sm_model = Model(inputs=[input_left, input_right], outputs=prediction)
    return output_siamese,input_left,input_right, sm_model
    
# def create_mlp_model(output_siamese_shape):

    # num_classes=71;
    # input_shape=Input((None,8192))
  
    
    # Create the model
    # model2 = Sequential()
    # model2.add(Dense(8192, input_shape=output_siamese_shape, activation='relu'))
    # model2.add(Dense(4096, input_shape=output_siamese_shape,activation='relu'))
    # model2.add(Dense(2048, activation='relu'))
    # model2.add(Dense(1024, activation='relu'))
    # model2.add(Dense(num_classes, activation='softmax'))
    
    # model2.summary()
    
    # out_siamese=Input(output_siamese_shape)
    # out = model2.output
    
    # return model2.input,out
    
def create_mlp(image_shape,dropout_rate):
    output_siamese,input_left,input_right, siamese_model = create_siamese_model(image_shape,
                                      dropout_rate)
    
    siamese_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
    siamese_model.fit(x_train,epochs=EPOCHS,steps_per_epoch=steps)
    
    num_classes=45;
    
    x = output_siamese
    x = Dense(4096, activation='relu')(x)
    x = Dense(2048, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(num_classes, activation='sigmoid')(x)                               
    #input_mlp,output_mlp= create_mlp_model(output_siamese.shape)
    #output_siamese=Input(output_siamese_shape)
    input_mlp=Input(output_siamese)
    mlp_model = Model(inputs=input_mlp, outputs=x)
    
    return mlp_model
    
mlp_model = create_mlp(image_shape,dropout_rate)
    
mlp_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
mlp_model.fit(x_train,epochs=EPOCHS,steps_per_epoch=steps)











