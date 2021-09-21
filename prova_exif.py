from re import I
from models import exif
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import Input
from keras import Model
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
from PIL.ExifTags import TAGS
from random import randint
import random
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def random_list(list):
    second_list = []
    for i in range(len(list)):
        if i % 300 == 0:
            second_list.append(list[i])
        else:
            second_list.append(random.choice(list))
    print("[INFO] Generated second list")
    return second_list
    
def labels(im1,im2):
    label=[]
    for i in range(len(im1)):
       if(im1[i]==im2[i]):
        label[i]=1
       else:
        label[i]=0;
    return label

def image_exif(im1,im2):

    # read the image data using PIL
    image1 = Image.open(im1)
    image2 = Image.open(im2)
    
    
    # extract EXIF data
    exifdata1 = image1.getexif()
    exifdata2 = image2.getexif()
    # iterating over all EXIF data fields
    exif1 = []
    exif2 = []
    for tag_id in exifdata1:
        # get the tag name, instead of human unreadable tag id
        tag = TAGS.get(tag_id, tag_id)
        data1 = exifdata1.get(tag_id)
        data2 = exifdata1.get(tag_id)
        exif1.append(data1)
        exif2.append(data2)    

    print("[INFO] Exif")
    return exif1,exif2

def datagenerator(images, labels, batchsize, mode="train"):
    ssad = 1
    while True:
        start = 0
        end = batchsize
        while start  < len(images):
            #if(len(images)-start < batchsize):
            #    break
            # load your images from numpy arrays or read from directory
            #else:
            x1 = images[start:end]
            #x2 = images2[start:end]
            y = np.array(labels[start:end])
            
            yield x1, y

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
    #x = Flatten(name=flatten_name)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(dropout_rate)(x)

    #x = Dense(512, activation='relu')(x)
    #x = Dropout(dropout_rate)(x)

    return x, model.input


def create_siamese_model(image_shape, dropout_rate):

    output_left, input_left = create_base_model(image_shape, dropout_rate)
    output_right, input_right = create_base_model(image_shape, dropout_rate, suffix="_2")
    #output = tf.concat([output_left,output_right],0)
    
    L1_layer = Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([output_left, output_right])
    L1_prediction = Dense(71, use_bias=True,
                          activation='sigmoid',
                          input_shape = image_shape,
                          kernel_initializer=RandomNormal(mean=0.0, stddev=0.001),
                          name='weighted-average')(L1_distance)
    prediction = Flatten()(L1_prediction)
    #prediction = Dropout(0.2)(L1_prediction)
    
    siamese_model = Model(inputs=[input_left, input_right], outputs= prediction)

    return siamese_model


siamese_model = create_siamese_model(image_shape=(128,128, 3),
                                         dropout_rate=0.2)

siamese_model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=0.000001),
                      metrics=['categorical_crossentropy', 'acc'])

with open("exif_lbl.txt", "rb") as fp:   #Picklingpickle.dump(l, fp)
	exif_lbl = pickle.load(fp)
fp.close()

#######################################################################################à
#crop images to 128x128
#######################################################################################à
list1,list2 = get_np_arrays('cropped_arrays.npy')



list3=random_list(list1)
#x_train = datagenerator(list1,exif_lbl,32)



imagexs = np.expand_dims(list1[0],axis=0)
imagexs2 = np.expand_dims(list2[0],axis=0)
exif1,exif2= image_exif('D02_img_orig_0001.jpg','D01_img_orig_0001.jpg') 

labels=[]
somma=0
#print(exif_lbl)
for i in range(len(exif_lbl)):
     for j in range(len(exif_lbl[0])):
         somma = exif_lbl[0][j]+somma
         if j % 15 == 0:
            somma=somma+randint(0, 1)
     labels.append(somma)
     somma=0

    
images1=[]
images2=[]

for i in range (len(exif_lbl[0])):
     images1.append(imagexs)
     images2.append(imagexs2)

im1 =cv2.imread('D01_img_orig_0001.jpg')
im2 =cv2.imread('D02_img_orig_0001.jpg')
#exif1,exif2=image_exif(im1,im2)
#print(exif1)
image1=tf.stack(images1,axis=0)
image2=tf.stack(images2,axis=0)

# tf.compat.v1.disable_eager_execution()
# im1= tf.compat.v1.placeholder(im1, [None, 128, 128, 3])
# im2  =  tf.compat.v1.placeholder(im2, [None, 128, 128, 3])
#label =  tf.compat.v1.placeholder(np.zeros(71), [None, 71])


siamese_model.fit(datagenerator((list1,list2),exif_lbl,batchsize=64, mode="train"),epochs=10)
