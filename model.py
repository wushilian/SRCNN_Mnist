import tensorflow as tf
import config as cfg
from sklearn.utils import shuffle
slim=tf.contrib.slim

class SRCNN:
    def __init__(self,sess):
        self.LR=tf.placeholder(tf.float32,[None,cfg.width,cfg.height,cfg.channel])#低分辨图片
        self.HR=tf.placeholder(tf.float32,[None,cfg.width,cfg.height,cfg.channel])#高分辨图片
        self.gen_HR=self.network()#CNN生成的高分辨图片
        self.sess=sess



    def network(self):
        conv1=slim.conv2d(self.LR,64,(9,9),scope='conv1')
        conv2 = slim.conv2d(conv1, 32, (1,1), scope='conv2')
        conv3 = slim.conv2d(conv2,cfg.channel, (5,5), scope='conv3')
        return conv3

    def train(self,LR_imgs,HR_imgs):
        saver = tf.train.Saver(max_to_keep=1)
        self.loss = tf.reduce_mean(tf.square(self.HR - self.gen_HR))  # MSE
        self.train_op = tf.train.AdamOptimizer(cfg.learning_rate).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())  # 初始化
        for i in range(cfg.epoch):
            LR_imgs,HR_imgs=shuffle(LR_imgs,HR_imgs)
            for j in range(int(HR_imgs.shape[0]/cfg.batch_size)):
                LR_batch,HR_batch=LR_imgs[j*cfg.batch_size:(j+1)*cfg.batch_size],HR_imgs[j*cfg.batch_size:(j+1)*cfg.batch_size]

                self.sess.run(self.train_op,feed_dict={self.LR:LR_batch,self.HR:HR_batch})
                if j%5==0:
                    print(i,self.sess.run(self.loss,feed_dict={self.LR:LR_batch,self.HR:HR_batch}))
            saver.save(self.sess,cfg.model_ckpt,global_step=i)


