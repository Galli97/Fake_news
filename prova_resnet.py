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
imagexs = np.expand_dims(list1[0],axis=0)
imagexs2 = np.expand_dims(list2[0],axis=0)
num_classes=71
# imagexs=tf.stack([imagexs,imagexs2],axis=0)
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
    x = Dense(128, activation='relu')(x)
    x = Flatten(name=flatten_name)(x)
    

    return x, model.input


def create_siamese_model(image_shape, dropout_rate):

    
    output_left, input_left = create_base_model(image_shape, dropout_rate)
    output_right, input_right = create_base_model(image_shape, dropout_rate, suffix="_2")
    
    output_siamese = tf.concat([output_left,output_right],1)
    
    siamese_model = Model(inputs=[input_left, input_right], outputs=output_siamese)

    return siamese_model,output_siamese
    
    
def create_mlp_model(output_siamese_shape):

    num_classes=71;
    #input_shape=Input((None,8192))
  
    
    # Create the model
    model2 = Sequential()
    #model2.add(Dense(8192, input_shape=output_siamese_shape, activation='relu'))
    model2.add(Dense(4096, input_shape=output_siamese_shape,activation='relu'))
    model2.add(Dense(2048, activation='relu'))
    model2.add(Dense(1024, activation='relu'))
    model2.add(Dense(num_classes, activation='softmax'))
    
    model2.summary()
    
    # out_siamese=Input(output_siamese_shape)
    out = model2.output
    
    return model2.input,out
    
def create_mlp(image_shape,dropout_rate):
    siamese_model, output_siamese = create_siamese_model(image_shape,
                                      dropout_rate)
                                      
    input_mlp,output_mlp= create_mlp_model(output_siamese.shape)
    #output_siamese=Input(output_siamese_shape)
    mlp_model = Model(inputs=input_mlp, outputs=output_mlp)
    
    return mlp_model
    


"""
siamese_model = create_siamese_model(image_shape=(128,128, 3),
                                         dropout_rate=0.2)
siamese_model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=0.0001),
                      metrics=['binary_crossentropy', 'acc'])
imagexs =cv2.imread('D01_img_orig_0001.jpg')[:,:,[2,1,0]]
imagexs = np.array(imagexs,np.float32)
imagexs = util.random_crop(imagexs,[128,128])
imagexs = np.expand_dims(imagexs,axis=0)
siamese_model.summary()
tmp1 = np.empty((5, 128, 128, 3), dtype=np.uint8)
for i in range(len(tmp1)):
    tmp1[i] = imagexs
x  = (tmp1,tmp1)
siamese_model.fit(x = (imagexs,imagexs),y=(imagexs),batch_size = 32,epochs=10)
                            #verbose=1,
                            #callbacks=[checkpoint, tensor_board_callback, lr_reducer, early_stopper, csv_logger],
                            #validation_data=(imagexs,imagexs))
                            #max_q_size=3)
#siamese_model.save('siamese_model.h5')
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
############################################################################################### FINE
"""


# siamese_model.compile(loss='binary_crossentropy',
                      # optimizer=Adam(lr=0.0001),
                      # metrics=['binary_crossentropy', 'acc'])
                      
# siamese_model.fit(x = (imagexs,imagexs2),y = output_siamese,epochs=10)

mlp_model=create_mlp(image_shape=(128,128,3),dropout_rate=0.2)

mlp_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


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

steps = len(list1)/EPOCHS


imagexs = np.expand_dims(list1[0],axis=0)
imagexs2 = np.expand_dims(list2[0],axis=0)
imagexs=tf.stack([imagexs,imagexs2],axis=0)

mlp_model.fit(x_train,epochs=EPOCHS,steps_per_epoch=steps)

# with open("exif_lbl.txt", "rb") as fp:   #Picklingpickle.dump(l, fp)
	# exif_lbl = pickle.load(fp)
# fp.close()

#######################################################################################à
#crop images to 128x128
#######################################################################################à
# list1,list2 = get_np_arrays('cropped_arrays.npy')


#siamese_model.fit_generator(datagenerator(list1,exif_lbl,32),steps_per_epoch=32,epochs=10,verbose=1)
#                            #callbacks=[checkpoint, tensor_board_callback, lr_reducer, early_stopper, csv_logger],
#                            #validation_data=x_train)
                            #max_q_size=3)
                            # 
# imagexs = np.expand_dims(list1[0],axis=0)
# imagexs2 = np.expand_dims(list2[0],axis=0)
#imagexs=tf.stack([imagexs,imagexs2],axis=0)
#label=np.zeros(len(exif_lbl));
# for i in range(len(exif_lbl)):
       # label[i]=[exif_lbl[i]]

    
#siamese_model.fit(x = (imagexs,imagexs2),y = imagexs2,epochs=10)