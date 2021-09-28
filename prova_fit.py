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

###########################################################################################################
#EXTRACTION#
###########################################################################################################
#extract exif data
#dict,image_list,dict_keys = extract_exif()

#############################################SAVE DICT##############################################
#with open("dict.pkl", "wb") as fp:   #Picklingpickle.dump(l, fp)#
#	pickle.dump(dict,fp)
#fp.close()

with open("dict.pkl", "rb") as fp:   #Picklingpickle.dump(l, fp)
	dict = pickle.load(fp)
fp.close()
#############################################SAVE IMAGE LIST##############################################
#with open("list_img.pkl", "wb") as fp:   #Picklingpickle.dump(l, fp)#
#	pickle.dump(image_list,fp)
#fp.close()

with open("list_img.pkl", "rb") as fp:   #Picklingpickle.dump(l, fp)
	image_list = pickle.load(fp)
fp.close()

#############################################SAVE DICT_KEYS##############################################
#with open("dict_keys.pkl", "wb") as fp:   #Picklingpickle.dump(l, fp)#
#	pickle.dump(dict_keys,fp)
#fp.close()

with open("dict_keys.pkl", "rb") as fp:   #Picklingpickle.dump(l, fp)
	dict_keys = pickle.load(fp)
fp.close()

#----------------------------------------------------------------------------------------------------------------------------------------

#generate second random list
second_image_list = random_list(image_list)

#generate lab els for each pair of images

exif_lbl = generate_label(dict_keys,image_list,second_image_list)

with open("exif_lbl.txt", "wb") as fp:   #Picklingpickle.dump(l, fp)#
	pickle.dump(exif_lbl,fp)
fp.close()

list1,list2 = cropping_list(image_list,second_image_list)

#with open("exif_lbl.txt", "rb") as fp:   #Picklingpickle.dump(l, fp)
#	exif_lbl = pickle.load(fp)
#fp.close()

for i in range(len(exif_lbl)):
    exif_lbl[i] = np.array(exif_lbl[i])
exif_lbl = np.array(exif_lbl)