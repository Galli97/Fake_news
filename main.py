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

#solver = initialize_exif()

solver.sess.run(tf.compat.v1.global_variables_initializer())
if solver.net.use_tf_threading:
    solver.coord = tf.train.Coordinator()
    solver.net.train_runner.start_p_threads(solver.sess)
    tf.train.start_queue_runners(sess=solver.sess, coord=solver.coord)

solver = initialize_exif(ckpt='/content/drive/MyDrive/ckpt/eval_100.ckpt', init=False)  #(ckpt='eval_160000.ckpt.data-00000-of-00001')


im1=util.random_crop(im1,[128,128])
im2=util.random_crop(im2,[128,128])

'''
data=[im1,im2,im3,im4]
exif_lbl = np.ones((lenght(data),83))
for i in lenght(data)
    exif_lbl[i] = 

'''
exif_lbl = np.ones((2,83))
exif_lbl[1] = np.random.randint(0,2,(1,83))
cls_lbl = np.zeros((1,1))
cls_lbl[0][0]=83


data=[im1,im2]
im1_merge = {'im_a':[im1,im2],'im_b':[im1,im1],'exif_lbl': exif_lbl,'cls_lbl': cls_lbl}
exif_solver.ExifSolver.setup_data(solver,data,im1_merge)
exif_solver.ExifSolver.train(solver)
"""
im = np.zeros((256, 256, 3))

bu = benchmark_utils.EfficientBenchmark(solver, nc, params, im, auto_close_sess=False, 
                                                     mirror_pred=False, dense_compute=False, stride=None, n_anchors=10,
                                                    patch_size=128, num_per_dim=30)
"""