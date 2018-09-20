import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
mb_size = 64
z_dim = 100
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
h_dim = 128
c = 0
lr = 1e-3


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def sample(mu, std):
    eps = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.square(std) * eps

# =============================== Q(z|X) ======================================

X = tf.placeholder(tf.float32, shape=[None, X_dim])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

Q_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
Q_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

Q_W2_mu = tf.Variable(xavier_init([h_dim, z_dim]))
Q_b2_mu = tf.Variable(tf.zeros(shape=[z_dim]))

Q_W2_sigma = tf.Variable(xavier_init([h_dim, z_dim]))
Q_b2_sigma = tf.Variable(tf.zeros(shape=[z_dim]))


def Q(X):
    h = tf.nn.relu(tf.matmul(X, Q_W1) + Q_b1)
    z_mu = tf.matmul(h, Q_W2_mu) + Q_b2_mu
    z_std = tf.nn.softplus(tf.matmul(h, Q_W2_sigma) + Q_b2_sigma)+1e-8
    return z_mu, z_std


# =============================== P(X|z) ======================================

P_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
P_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

P_W2_mu = tf.Variable(xavier_init([h_dim, X_dim]))
P_b2_mu = tf.Variable(tf.zeros(shape=[X_dim]))

P_W2_sigma = tf.Variable(xavier_init([h_dim, X_dim]))
P_b2_sigma = tf.Variable(tf.zeros(shape=[X_dim]))

def P(z):
    h = tf.nn.relu(tf.matmul(z, P_W1) + P_b1)
    x_mu = tf.matmul(h, P_W2_mu) + P_b2_mu
    x_std = tf.nn.softplus(tf.matmul(h, P_W2_sigma) + P_b2_sigma) + 1e-8
    return x_mu, x_std


# =============================== TRAINING ====================================

z_mu, z_std = Q(X)
z_sample = sample(z_mu, z_std)
x_mu, x_std = P(z_sample)
X_pre=sample(x_mu,x_std)

# Sampling from random z
X_samples= sample(P(z)[0],P(z)[1])


# E[log P(X|z)]
recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=X_pre, labels=X), 1)

# D_KL(Q(z|X) || P(z)); calculate in closed form as both dist. are Gaussian
kl = -0.5 * tf.reduce_sum(1-tf.square(z_mu) -tf.square(z_std) + tf.log(tf.square(z_std)+lr) , 1)
# kl2= -0.5 * tf.reduce_sum(1-tf.square(x_mu) -tf.square(x_std) + tf.log(tf.square(x_std)+lr) , 1)
elbo = tf.reduce_sum(recon_loss+kl)

solver = tf.train.GradientDescentOptimizer(lr).minimize(-elbo)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out2/'):
    os.makedirs('out2/')

i = 0

for it in range(1000000):
    X_mb, _ = mnist.train.next_batch(mb_size)

    _, loss = sess.run([solver, elbo], feed_dict={X: X_mb})

    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('Loss: {:.4}'.format(loss))
        print()

        samples = sess.run(X_samples, feed_dict={z: np.random.randn(16, z_dim)})

        fig = plot(samples)
        plt.savefig('out2/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)
