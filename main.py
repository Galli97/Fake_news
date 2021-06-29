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
print("---------------------------------------------------------------------")
print(im1.shape)
print("---------------------------------------------------------------------")
solver = initialize_exif()
solver.sess.run(tf.compat.v1.global_variables_initializer())
if solver.net.use_tf_threading:
    solver.coord = tf.train.Coordinator()
    solver.net.train_runner.start_p_threads(solver.sess)
    tf.train.start_queue_runners(sess=solver.sess, coord=solver.coord)

im1=util.random_crop(im1,[128,128])
im1_merge = {'im_a':[im1,im1,im1,im1,im1,im1],'im_b':[im1,im1,im1,im1,im1,im1]}
exif_solver.ExifSolver.setup_data(solver,im1,im1_merge)
exif_solver.ExifSolver.train(solver)
"""
im = np.zeros((256, 256, 3))

bu = benchmark_utils.EfficientBenchmark(solver, nc, params, im, auto_close_sess=False, 
                                                     mirror_pred=False, dense_compute=False, stride=None, n_anchors=10,
                                                    patch_size=128, num_per_dim=30)
"""