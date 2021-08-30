from re import I
from models import exif
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import Input
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
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

def create_base_model(image_shape, dropout_rate, suffix=''):
    left_input = Input(image_shape)

    model = ResNet50(include_top=False, weights='imagenet', input_tensor=left_input, pooling=None)
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1]._outbound_nodes = []

    for layer in model.layers:
        layer._name = layer.name + str(suffix)
        layer._trainable = False

    flatten_name = 'flatten' + str(suffix)

    x = model.output
    x = Flatten(name=flatten_name)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
   
    return model


def create_siamese_model(image_shape):
    
    image_shape1 = Input(image_shape)
    model = Sequential()
    model.add(Conv2D(4096, (10,10), activation='relu', input_shape=image_shape1))
    model.add(Flatten())
    model.add(Conv2D(2048, (7,7), activation='relu'))
    model.add(Flatten())
    model.add(Conv2D(1024, (4,4), activation='relu'))
   
    return model
    

encoded_l = create_base_model(image_shape=(64, 64, 3),dropout_rate=0.2)
encoded_r = create_base_model(image_shape=(64, 64, 3),dropout_rate=0.2)

input_mlp = tf.concat([encoded_l,encoded_r],0)
input_mlp=tf.convert_to_tensor(input_mlp)
siamese_model = create_siamese_model(input_mlp)
                                         # dropout_rate=0.2)
    # output = create_base_model(image_shape, dropout_rate)
   
    # print("------------------------------------------------------------------------------")
    # print(output_left)
    # print("------------------------------------------------------------------------------")
    # output = tf.concat([output_left,output_right],0)
    
    
    # L1_layer = Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))
    # L1_distance = L1_layer([output_left, output_right])
    # L1_prediction = Dense(1, use_bias=True,
                          # activation='sigmoid',
                          # input_shape = image_shape,
                          # kernel_initializer=RandomNormal(mean=0.0, stddev=0.001),
                          # name='weighted-average')(L1_distance)
    # prediction = Dropout(0.2)(L1_prediction)

    # siamese_model = Model(inputs=[input_left, input_right], outputs=prediction)

    # return siamese_model

# siamese_model = create_base_model(image_shape=(64, 64, 3),
                                         # dropout_rate=0.2)



imagexs =cv2.imread('D01_img_orig_0001.jpg')[:,:,[2,1,0]]

imagexs = np.array(imagexs,np.float32)
imagexs = util.random_crop(imagexs,[64,64])
imagexs = np.expand_dims(imagexs,axis=0)

siamese_model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=0.0001),
                      metrics=['binary_crossentropy', 'acc'])
siamese_model.fit(x=(imagexs,imagexs),y=(imagexs),batch_size = 32,
                            epochs=10)

siamese_model.save('siamese_model.h5')


"""
# and the my prediction
siamese_net = load_model('siamese_model.h5', custom_objects={"tf": tf})
X_1 = [image, ] * len(markers)
batch = [markers, X_1]
result = siamese_net.predict_on_batch(batch)
# I've tried also to check identical images 
markers = [image]
X_1 = [image, ] * len(markers)
batch = [markers, X_1]
result = siamese_net.predict_on_batch(batch)
"""