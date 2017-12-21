from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import config as cfg
from scipy.ndimage import zoom
import cv2
import os
import model
import tensorflow as tf
from matplotlib import  pyplot as plt
def get_mnist():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    HR_img=np.reshape(mnist.train.images[0:200],(-1,28,28,1))
    LR_img=[]
    for i in range(len(HR_img)):
        _img=zoom(np.squeeze(HR_img[i]),cfg.s)
        LR_img.append(cv2.resize(_img,(cfg.width,cfg.height))[:,:,np.newaxis])
    return np.array(LR_img),HR_img
LR_img,HR_img=get_mnist()
sess=tf.Session()
srcnn=model.SRCNN(sess)
restorer=tf.train.Saver()
restorer.restore(sess,tf.train.latest_checkpoint('weightfile'))
def save_SR(LR_img):
    SR_img=sess.run(srcnn.gen_HR,feed_dict={srcnn.LR:LR_img})
    z = np.concatenate((LR_img,SR_img), 2)
    for i in range(z.shape[0]):
        cv2.imwrite(os.path.join('SR',str(i)+'.jpg'),z[i]*256)
save_SR(LR_img)
