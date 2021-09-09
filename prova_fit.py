from re import I
from models import exif
from PIL import Image
import tensorflow as tf
from tensorflow import keras
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
from sklearn import preprocessing
import time
from random import randint

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
    x = Flatten(name=flatten_name)(x)
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
    L1_prediction = Dense(1, use_bias=True,
                          activation='sigmoid',
                          input_shape = image_shape,
                          kernel_initializer=RandomNormal(mean=0.0, stddev=0.001),
                          name='weighted-average')(L1_distance)

    prediction = Dropout(0.2)(L1_prediction)

    siamese_model = Model(inputs=[input_left, input_right], outputs=prediction)

    return siamese_model
	
siamese_model = create_siamese_model(image_shape=(128,128, 3),
                                         dropout_rate=0.2)

siamese_model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=0.0001),
                      metrics=['categorical_crossentropy', 'acc'])

with open("exif_lbl.txt", "rb") as fp:   #Picklingpickle.dump(l, fp)
	exif_lbl = pickle.load(fp)
fp.close()

#######################################################################################à
#crop images to 128x128
#######################################################################################à
list1,list2 = get_np_arrays('cropped_arrays.npy')

labels=[]
somma=0
for i in range(len(exif_lbl)):
     for j in range(len(exif_lbl[0])):
         somma = exif_lbl[0][j]+somma
         if j % 64 == 0:
            somma=somma+randint(0, 5)
     labels.append(somma)
     somma=0
print(labels)
# Instantiate an optimizer.
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-5)
# Instantiate a loss function.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

#####
# Prepare the metrics.
# train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
# val_acc_metric = keras.metrics.SparseCategoricalAccuracy()
########

# Prepare the training dataset.
batch_size = 64
x1_train=list1
x2_train=list2
y_train=labels
#y_train=np.reshape(y_train, (-1, 273421))
#x1_train = np.reshape(x1_train, (-1, 16384,3)) #(128x128x3)
#x2_train = np.reshape(x2_train, (-1, 16384,3))#(128x128x3)
train_dataset = tf.data.Dataset.from_tensor_slices(((x1_train,x2_train), y_train))

##########
# #Prepare the validation dataset.
## Reserve 10,000 samples for validation.
# x1_val = x1_train[-1000:]
# x2_val = x2_train[-1000:]
# y_val = y_train[-1000:]
# x1_train = x1_train[:-1000]
# x2_train = x2_train[:-1000]
# y_train = y_train[:-1000]
# val_dataset = tf.data.Dataset.from_tensor_slices(((x1_val,x2_val), y_val))
# val_dataset = val_dataset.batch(64)
##########
epochs = 2
g=0
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    ##
    # start_time = time.time()
    ##
    g=0
    # Iterate over the batches of the dataset.
    for step, ((x1_batch_train,x2_batch_train), y_batch_train) in enumerate(train_dataset):
        g=g+1
        if (g<500):
            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:

                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                x1_batch_train = np.expand_dims(x1_batch_train,axis=0)
                x2_batch_train = np.expand_dims(x2_batch_train,axis=0)

                logits = siamese_model((x1_batch_train,x2_batch_train), training=True)  # Logits for this minibatch

                # Compute the loss value for this minibatch.
                y_batch_train = np.expand_dims(y_batch_train,axis=0)
                loss_value = loss_fn(y_batch_train, logits)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, siamese_model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, siamese_model.trainable_weights))
            
            ####
            # Update training metric.
            # train_acc_metric.update_state(y_batch_train, logits)
            ####

            # Log every 200 batches.
            if step % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %s samples" % ((step + 1) * 64))
    ############   
    # Display metrics at the end of each epoch.
    # train_acc = train_acc_metric.result()
    # print("Training acc over epoch: %.4f" % (float(train_acc),))

    # Reset training metrics at the end of each epoch
    # train_acc_metric.reset_states()
    
    # Run a validation loop at the end of each epoch.
    # for (x1_batch_val,x2_batch_val), y_batch_val in val_dataset:
        # val_logits = siamese_model((x1_batch_val,x2_batch_val), training=False)
        # Update val metrics
        # val_acc_metric.update_state(y_batch_val, val_logits)
    # val_acc = val_acc_metric.result()
    # val_acc_metric.reset_states()
    # print("Validation acc: %.4f" % (float(val_acc),))
    # print("Time taken: %.2fs" % (time.time() - start_time))
    ###########






















# imagexs = np.expand_dims(list1[0],axis=0)
# imagexs2 = np.expand_dims(list2[0],axis=0)


# y=np.reshape(exif_lbl[0],(1,71))
# y = np.array(y)
# print(y.shape)

# x_train = datagenerator(list1,list2,exif_lbl,32)
# siamese_model.fit(x_train,epochs=10)