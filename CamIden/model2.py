import tensorflow as tf
from Dataload import dataload
import numpy as np
import os

tf.set_random_seed(1)
np.random.seed(1)
BATCH_SIZE = 5
LR = 1e-4              # learning rate

save_path='model/'
model_name='cnn_model.ckpt'
filename='CamData.npy'
filename2='CamLabel.npy'
cam_image=dataload(filename,filename2)
test_x,test_y=cam_image.test_x_y()
if not os.path.exists(save_path):
    os.makedirs(save_path)

with tf.variable_scope('Input'):
    tf_x = tf.placeholder(tf.float32, [None, 128*128*3],name='x')
    tf_y = tf.placeholder(tf.int32, [None, 1],name='y')            # input y
             # (batch, height, width, channel)
tf_dropout = tf.placeholder(tf.bool, None) # dropout
tf_bn=tf.placeholder(tf.bool, None)
image = tf.reshape(tf_x,[-1,128,128,3])
image = tf.layers.batch_normalization(image,training=tf_bn)
# CNN
with tf.name_scope('Con_Layer1'):
    conv1 = tf.layers.conv2d(   # shape (128, 128, 3)
            inputs=image,
            filters=12,
            kernel_size=5,
            strides=2,
            padding='same',
            activation=None
        )           # -> (64, 64, 12)
    conv1 = tf.layers.batch_normalization(conv1,training=tf_bn)
    conv1 = tf.nn.relu(conv1)
with tf.name_scope('Con_Layer2'):
    conv2 = tf.layers.conv2d(conv1, 12, 5, 2, 'same', activation=None,)    # -> (32, 32, 12)
    conv2 = tf.layers.batch_normalization(conv2,training=tf_bn)
    conv2 = tf.nn.relu(conv2)
with tf.name_scope('Pool_Layer'):
    pool = tf.layers.max_pooling2d(conv2, 2, 2,)    # -> (16, 16, 12)
    pool =tf.layers.batch_normalization(pool,training=tf_bn)
with tf.name_scope('DeCon_Layer1'):
    deconv1=tf.layers.conv2d_transpose(pool,6,5,2,'same') # -> (32, 32, 6)
    deconv1 = tf.nn.relu(deconv1)
with tf.name_scope('DeCon_Layer2'):
    deconv2=tf.layers.conv2d_transpose(deconv1,3,5,4,'same') # -> (128, 128, 3)
    deconv2 = tf.nn.relu(deconv2)
flat = tf.reshape(deconv2, [-1, 128*128*3])          # -> (128*128*3, )
flat = tf.layers.dropout(flat, rate=0.8, training=tf_dropout)
with tf.name_scope('Out_layer'):  # out layer
    output = tf.layers.dense(flat, 128*128*3,activation=tf.nn.sigmoid)

tf.summary.histogram('Con_Layer1', conv1)
tf.summary.histogram('Con_Layer2', conv2)
tf.summary.histogram('Pool_Layer', pool)
tf.summary.histogram('DeCon_layer1', deconv1)
tf.summary.histogram('DeCon_layer2', deconv2)
tf.summary.histogram('Out_layer', output)
# loss
loss = tf.losses.mean_squared_error(labels=tf_x,predictions=output)  # compute cost
with tf.name_scope('Train'):
    train_op = tf.train.AdamOptimizer(LR).minimize(loss)
tf.summary.scalar('loss', loss)

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())  # the local var is for accuracy_op
sess.run(init_op)  # initialize var in graph
# tensorboard data
writer = tf.summary.FileWriter('./logs', sess.graph)  # write to file
merge_op = tf.summary.merge_all()  # operation to merge all summary


def cnn_train(t_x,t_y):
    for step in range(2000):
        b_x, b_y = [t_x,t_y]
        _, loss_ = sess.run([train_op, loss], {tf_x: b_x,tf_dropout:True,tf_bn:True})
        print('Step:', step , '| loss: %.2f' %loss_)
    sess.close()

def cnn_pre(t_x):

    pre_y=tf.argmax(output,axis=1)

    save_path_full = os.path.join(save_path, model_name)


    cls = sess.run(pre_y, {tf_x: t_x, tf_dropout: True, tf_bn: True})
    return cls
    # acc=sess.run(accuracy,{tf_x:t_x,tf_y:test_y,tf_dropout:True,tf_bn:True})
    # return acc


cnn_train(cam_image.train_next_batch(BATCH_SIZE))