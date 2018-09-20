import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow.contrib.slim as slim
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
mb_size = 64
z_dim = 10
x_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
w_dim=28*28
h_dim = 20
lr = 1e-3
x = tf.placeholder(tf.float32, shape=[None, w_dim])
index=0
distributions=tf.distributions
sess = tf.Session()



def plot(samples,x,y):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(x, y), cmap='Greys_r')

    return fig


# window sliding
def window_sliding(X,w_dim,index):
    x_dim=X.shape[1]
    if w_dim>x_dim:
        return X
    x_w=X[:,index*w_dim:index*w_dim+w_dim]
    return x_w

# =============================== Encoder Q(z|X) ======================================

def Encoder(x):

    with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
        net = slim.fully_connected(x, h_dim)
        net = slim.fully_connected(net, h_dim)
        net = slim.fully_connected(net, h_dim)
        gaussian_params = slim.fully_connected(net, z_dim * 2, activation_fn=None)
        # The mean parameter is unconstrained
        mu = gaussian_params[:, :z_dim]
        # The standard deviation must be positive. Parametrize with a softplus
        sigma = tf.nn.softplus(gaussian_params[:, z_dim:])+1e-8
        return mu, sigma

# =============================== Decoder P(X|z) ======================================

def Decoder(z):

    with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
        net = slim.fully_connected(z, h_dim)
        net = slim.fully_connected(net, h_dim)
        net = slim.fully_connected(net, h_dim)
        gaussian_params = slim.fully_connected(net, w_dim*2, activation_fn=None)
        # The mean parameter is unconstrained
        mu = gaussian_params[:, :w_dim]
        # The standard deviation must be positive. Parametrize with a softplus
        sigma = tf.nn.softplus(gaussian_params[:, w_dim:]) + 1e-8
        return mu, sigma


# =============================== TRAINING ====================================

def train():
    # Train a Variational Autoencoder on MNIST
    # Input placeholders

    with tf.variable_scope('variational'):
        q_mu, q_sigma = Encoder(x)
        # The variational distribution is a Normal with mean and standard
        # deviation given by the inference network
        q_z = distributions.Normal(loc=q_mu, scale=tf.square(q_sigma))
        assert q_z.reparameterization_type == distributions.FULLY_REPARAMETERIZED

    with tf.variable_scope('model'):
        # generative network
        p_mu,p_sigma = Decoder(q_z.sample())
        p_x =distributions.Normal(p_mu,tf.square(p_sigma))
        assert p_x.reparameterization_type == distributions.FULLY_REPARAMETERIZED
        posterior_predictive_samples = p_x.sample()
        tf.summary.image('posterior_predictive',
                         tf.cast(posterior_predictive_samples, tf.float32))

        # Take samples from the prior
    with tf.variable_scope('model', reuse=True):
        p_z = distributions.Normal(loc=np.zeros(z_dim, dtype=np.float32),
                                       scale=np.square(np.ones(z_dim, dtype=np.float32)))
        p_z_sample = p_z.sample(1)
        p_x_z_mu,p_x_z_sigma = Decoder(p_z_sample)
        prior_predictive = distributions.Normal(p_x_z_mu,tf.square(p_x_z_sigma))
        prior_predictive_samples = prior_predictive.sample()
        tf.summary.image('prior_predictive',
                             tf.cast(prior_predictive_samples, tf.float32))
        # Take samples from the prior with a placeholder
    with tf.variable_scope('model', reuse=True):
        z_input = tf.placeholder(tf.float32, [None, z_dim])
        p_x_given_z_mu,p_x_given_z_sigma = Decoder(z_input)
        prior_predictive_inp = distributions.Normal(p_x_given_z_mu,tf.square(p_x_given_z_sigma))
        prior_predictive_inp_sample = prior_predictive_inp.sample()


    # Build the evidence lower bound (ELBO) or the negative loss

    kl = tf.reduce_sum(distributions.kl_divergence(q_z, p_z), 1)
    expected_log_likelihood = tf.reduce_sum(p_x.log_prob(x),1)

    elbo = tf.reduce_sum(expected_log_likelihood-kl)
    # elbo = tf.log(posterior_predictive_samples)
    acc=tf.metrics.auc(labels=x,predictions=posterior_predictive_samples)
    train_op = tf.train.RMSPropOptimizer(lr).minimize(-elbo)

    # Merge all the summaries
    summary_op = tf.summary.merge_all()

    sess.run(tf.global_variables_initializer())
    if not os.path.exists('out/'):
        os.makedirs('out/')
    train_writer = tf.summary.FileWriter("./logs", sess.graph)
    for it in range(500000):
        X_mb, _ = mnist.train.next_batch(mb_size)
        index=0

        while index<X_mb.shape[1]//w_dim:
            x_w=window_sliding(X_mb,w_dim,index)
            _, loss_ = sess.run([train_op, elbo], feed_dict={x: x_w})
            index += 1
            if it % 1000 == 0:
                print('Iter: {}'.format(it)+'  Sliding_Index: {}'.format(index)+'  Loss: {:.4}'. format(loss_))



def predicte(x_test):

    q_mu, q_sigma = Encoder(x)
    q_z = distributions.Normal(loc=q_mu, scale=tf.square(q_sigma))
    p_mu,p_sigma = Decoder(q_z.sample())
    p_x =distributions.Normal(p_mu,tf.square(p_sigma))
    posterior_predictive_samples = p_x.sample()
    samples = sess.run(posterior_predictive_samples, feed_dict={x: x_test})
    return samples


def data_standardization(dataset):
    for data in dataset:
        if "null" in data:
            for index,value in enumerate(data):
                if value=="null":
                    data[index]=0
    return np.asarray(dataset,dtype=np.float32)



