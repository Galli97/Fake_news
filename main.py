from models import exif
import random
from models.exif import exif_solver,exif_net
from load_models import initialize_exif
from extract_exif import extract_exif, random_list,generate_label,cropping_list
from lib.utils import benchmark_utils, util,io
from demo import Demo
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

#extract exif data
dict,image_list,dict_keys = extract_exif()

#generate second random list
second_image_list = random_list(image_list)

#generate labels for each pair of images

exif_lbl = generate_label(dict_keys,image_list,image_list)

#crop images to 128x128

list1,list2 = cropping_list(image_list,second_image_list)


#start initialization

solver = initialize_exif()
solver.sess.run(tf.compat.v1.global_variables_initializer())
if solver.net.use_tf_threading:
    solver.coord = tf.train.Coordinator()
    solver.net.train_runner.start_p_threads(solver.sess)
    tf.train.start_queue_runners(sess=solver.sess, coord=solver.coord)

cls_lbl = np.ones((1,1))
cls_lbl[0][0] = len(dict_keys)


im1_merge = {'im_a':list1,'im_b':list2,'exif_lbl': exif_lbl,'cls_lbl': cls_lbl}
exif_solver.ExifSolver.setup_data(solver,list1,im1_merge)
exif_solver.ExifSolver.train(solver)


'''
from models.exif import exif_solver,exif_net
from load_models import initialize_exif
from extract_exif import extract_exif
from lib.utils import benchmark_utils, util,io
from demo import Demo
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
#dict = extract_exif()
#salvati = dict.keys()
#print(salvati)
im1 = cv2.imread("D01_img_orig_0001.jpg")[:,:,[2,1,0]]
im2 = cv2.imread("D02_img_orig_0001.jpg")[:,:,[2,1,0]]
print("---------------------------------------------------------------------")
print(im1.shape)
print(im2.shape)
print("---------------------------------------------------------------------")

solver = initialize_exif()

solver.sess.run(tf.compat.v1.global_variables_initializer())

if solver.net.use_tf_threading:
    solver.coord = tf.train.Coordinator()
    solver.net.train_runner.start_p_threads(solver.sess)
    tf.train.start_queue_runners(sess=solver.sess, coord=solver.coord)
    solver.net.use_tf_threading=False

  
    

im1=util.random_crop(im1,[128,128])
im2=util.random_crop(im2,[128,128])

exif_lbl = np.ones((2,83))
exif_lbl[1] = np.random.randint(0,2,(1,83))
cls_lbl = np.zeros((1,1))
cls_lbl[0][0]=83


data=[im1,im2]
im1_merge = {'im_a':[im1,im2],'im_b':[im1,im1],'exif_lbl': exif_lbl,'cls_lbl': cls_lbl}
exif_solver.ExifSolver.setup_data(solver,data,im1_merge)
exif_solver.ExifSolver.train(solver)

#ckpt='/content/drive/MyDrive/ckpt/eval_100.ckpt'
#solver = exif_solver.initialize({'checkpoint':ckpt,  #(ckpt='eval_160000.ckpt.data-00000-of-00001')
#                                     'use_exif_summary':True, ###era false
 #                                    'init_summary':True,
  #                                   'exp_name':'eval'})
'''