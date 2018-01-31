import tensorflow as tf
from Dataload import dataload
import numpy as np
import os



class autoCnn:

    def __init__(self, bath_size,w,h,save_path,model_name):
        tf.set_random_seed(1)
        np.random.seed(1)
        self.BATCH_SIZE = bath_size
        self.LR = 1e-4
        self.save_path = save_path
        self.model_name = model_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        with tf.variable_scope('Input'):
            self.tf_x = tf.placeholder(tf.float32, [None, w * h * 1], name='x')
        self.tf_dropout = tf.placeholder(tf.bool, None)  # dropout
        self.tf_bn = tf.placeholder(tf.bool, None)
        self.sess = tf.Session()
        # self.saver = tf.train.Saver()
        self.save_path_full = os.path.join(save_path, model_name)

    def load(self,path):
        self.data=dataload(path)


    def model(self):
        image = tf.reshape(self.tf_x, [-1, 128, 128, 1])
        image = tf.layers.batch_normalization(image, training=self.tf_bn)
        # CNN
        with tf.name_scope('Con_Layer1'):
            conv1 = tf.layers.conv2d(  # shape (128, 128, 3)
                inputs=image,
                filters=12,
                kernel_size=5,
                strides=2,
                padding='same',
                activation=None
            )  # -> (64, 64, 12)
            conv1 = tf.layers.batch_normalization(conv1, training=self.tf_bn)
            conv1 = tf.nn.relu(conv1)
        with tf.name_scope('Con_Layer2'):
            conv2 = tf.layers.conv2d(conv1, 12, 5, 2, 'same', activation=None, )  # -> (32, 32, 12)
            conv2 = tf.layers.batch_normalization(conv2, training=self.tf_bn)
            conv2 = tf.nn.relu(conv2)
        with tf.name_scope('Pool_Layer'):
            pool = tf.layers.max_pooling2d(conv2, 2, 2, )  # -> (16, 16, 12)
            pool = tf.layers.batch_normalization(pool, training=self.tf_bn)
        with tf.name_scope('DeCon_Layer1'):
            deconv1 = tf.layers.conv2d_transpose(pool, 6, 5, 2, 'same')  # -> (32, 32, 6)
            deconv1 = tf.nn.relu(deconv1)
        with tf.name_scope('DeCon_Layer2'):
            deconv2 = tf.layers.conv2d_transpose(deconv1, 3, 5, 2, 'same')  # -> (128, 128, 3)
            deconv2 = tf.nn.relu(deconv2)
        flat = tf.reshape(deconv2, [-1, 64 * 64 * 3])  # -> (128*128*3, )
        flat = tf.layers.dropout(flat, rate=0.8, training=self.tf_dropout)
        with tf.name_scope('Out_layer'):  # out layer
            self.output = tf.layers.dense(flat, 128 * 128 * 1, activation=tf.nn.sigmoid)
        tf.summary.histogram('Con_Layer1', conv1)
        tf.summary.histogram('Con_Layer2', conv2)
        tf.summary.histogram('Pool_Layer', pool)
        tf.summary.histogram('DeCon_layer1', deconv1)
        tf.summary.histogram('DeCon_layer2', deconv2)
        tf.summary.histogram('Out_layer', self.output)
        self.saver = tf.train.Saver()

    def train(self):

        loss = tf.losses.mean_squared_error(labels=self.tf_x, predictions=self.output)  # compute cost
        with tf.name_scope('Train'):
            train_op = tf.train.AdamOptimizer(self.LR).minimize(loss)
        tf.summary.scalar('loss', loss)
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())  # the local var is for accuracy_op
        self.sess.run(init_op)  # initialize var in graph
        writer = tf.summary.FileWriter('./logs', self.sess.graph)  # write to file
        merge_op = tf.summary.merge_all()  # operation to merge all summary

        for step in range(600):
            b_x = self.data.train_next_batch(self.BATCH_SIZE)
            _,loss_ = self.sess.run([train_op, loss], {self.tf_x: b_x,self.tf_dropout: False, self.tf_bn: True})
            print('Step:', step, '| loss: %.2f' % loss_)
        self.saver.save(self.sess, self.save_path_full)
        self.sess.close()

    def pre(self,t_x):
        save_path_full = os.path.join(self.save_path, self.model_name)
        self.saver.restore(self.sess, save_path_full)
        cls = self.sess.run(self.output, {self.tf_x: t_x, self.tf_dropout:False,self.tf_bn: True})
        return cls