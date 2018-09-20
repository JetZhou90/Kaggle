import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec




class VAE:

    def __init__(self,x,h_dim,w_dim,z_dim,save_path,model_name,epoch,bath_size=0):
        self.BATCH_SIZE = bath_size
        self.epoch=epoch
        self.lr = 1e-4
        self.x=self.data_standardization(x)
        self.x_dim = x.shape[1]
        self.w_dim=w_dim
        with tf.variable_scope('Input'):
            self.tf_x = tf.placeholder(tf.float32, [None, self.w_dim], name='x')
        self.h_dim=h_dim
        self.z_dim=z_dim
        self.distributions = tf.distributions
        self.save_path = save_path
        self.model_name = model_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.sess = tf.Session()
        # self.saver = tf.train.Saver()
        self.save_path_full = os.path.join(save_path, model_name)
        self.Model()
        self.index=0

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

    # window sliding
    def window_sliding(self,x,w_dim,index):
        if w_dim > x.shape[1]:
            return x
        x_w = x[:, index * w_dim:index * w_dim + w_dim]
        return x_w

    # =============================== Encoder Q(z|X) ======================================

    def Encoder(self,x):

        with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
            net = slim.fully_connected(x, self.h_dim)
            net = slim.fully_connected(net, self.h_dim)
            net = slim.fully_connected(net, self.h_dim)
            gaussian_params = slim.fully_connected(net, self.z_dim * 2, activation_fn=None)
            # The mean parameter is unconstrained
            mu = gaussian_params[:, :self.z_dim]
            # The standard deviation must be positive. Parametrize with a softplus
            sigma = tf.nn.softplus(gaussian_params[:, self.z_dim:]) + 1e-8
            return mu, sigma

    # =============================== Decoder P(X|z) ======================================

    def Decoder(self,z):

        with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
            net = slim.fully_connected(z, self.h_dim)
            net = slim.fully_connected(net, self.h_dim)
            net = slim.fully_connected(net, self.h_dim)
            gaussian_params = slim.fully_connected(net, self.w_dim * 2, activation_fn=None)
            # The mean parameter is unconstrained
            mu = gaussian_params[:, :self.w_dim]
            # The standard deviation must be positive. Parametrize with a softplus
            sigma = tf.nn.softplus(gaussian_params[:, self.w_dim:]) + 1e-8
            return mu, sigma

    def Model(self):

        # Train a Variational Autoencoder on MNIST
        # Input placeholders

        with tf.variable_scope('variational'):
            q_mu, q_sigma = self.Encoder(self.tf_x)
            # The variational distribution is a Normal with mean and standard
            # deviation given by the inference network
            self.q_z = self.distributions.Normal(loc=q_mu, scale=tf.square(q_sigma))
            assert self.q_z.reparameterization_type == self.distributions.FULLY_REPARAMETERIZED

        with tf.variable_scope('model'):
            # generative network
            p_mu, p_sigma = self.Decoder(self.q_z.sample())
            self.p_x = self.distributions.Normal(p_mu, tf.square(p_sigma))
            assert self.p_x.reparameterization_type == self.distributions.FULLY_REPARAMETERIZED
            self.posterior_predictive_samples = self.p_x.sample()
            tf.summary.image('posterior_predictive',
                             tf.cast(self.posterior_predictive_samples, tf.float32))

            # Take samples from the prior
        with tf.variable_scope('model', reuse=True):
            self.p_z = self.distributions.Normal(loc=np.zeros(self.z_dim, dtype=np.float32),
                                       scale=np.square(np.ones(self.z_dim, dtype=np.float32)))
            self.p_z_sample = self.p_z.sample(1)
            p_x_z_mu, p_x_z_sigma = self.Decoder(self.p_z_sample)
            self.prior_predictive = self.distributions.Normal(p_x_z_mu, tf.square(p_x_z_sigma))
            self.prior_predictive_samples = self.prior_predictive.sample()
            tf.summary.image('prior_predictive',
                             tf.cast(self.prior_predictive_samples, tf.float32))
            # Take samples from the prior with a placeholder
        with tf.variable_scope('model', reuse=True):
            self.z_input = tf.placeholder(tf.float32, [None, self.z_dim])
            p_x_given_z_mu, p_x_given_z_sigma = self.Decoder(self.z_input)
            self.prior_predictive_inp = self.distributions.Normal(p_x_given_z_mu, tf.square(p_x_given_z_sigma))
            self.prior_predictive_inp_sample = self.prior_predictive_inp.sample()

        self.saver = tf.train.Saver()

    def Train(self):

        kl = tf.reduce_sum(self.distributions.kl_divergence(self.q_z, self.p_z), 1)
        expected_log_likelihood = tf.reduce_sum(self.p_x.log_prob(self.tf_x), 1)
        elbo = tf.reduce_sum(expected_log_likelihood - kl)
        # elbo = tf.log(posterior_predictive_samples)
        train_op = tf.train.RMSPropOptimizer(self.lr).minimize(-elbo)

        # Merge all the summaries
        summary_op = tf.summary.merge_all()
        self.sess.run(tf.global_variables_initializer())
        if not os.path.exists('out/'):
            os.makedirs('out/')
        train_writer = tf.summary.FileWriter("./logs", self.sess.graph)
        try:
            save_path_full = os.path.join(self.save_path, self.model_name)
            self.saver.restore(self.sess, save_path_full)
        except:
            pass
        for it in range(self.epoch):
            while self.index < self.x_dim // self.w_dim:
                x_w = self.window_sliding(self.x,self.w_dim,self.index)
                self.sess.run(train_op,feed_dict={self.tf_x: x_w})
                loss_ = self.sess.run(elbo, feed_dict={self.tf_x: x_w})
                self.index += 1
                if it % 10 == 0:
                    print('Iter: {}'.format(it) + '  Sliding_Index: {}'.format(self.index) + '  Loss: {:.4}'.format(loss_))
            self.index=0
            self.saver.save(self.sess, self.save_path_full)
        self.sess.close()

    def Predicte(self,x_test):
        results=np.empty([x_test.shape[0],1],dtype=np.float32)
        save_path_full = os.path.join(self.save_path, self.model_name)
        self.saver.restore(self.sess, save_path_full)

        while self.index < self.x.shape[1] // self.w_dim:
            x_w = self.window_sliding(x_test, self.w_dim,self.index)
            samples = self.sess.run(self.posterior_predictive_samples, feed_dict={self.tf_x: x_w})
            results=np.concatenate([results,samples],1)
        self.index=0
        return results[:,1:-1]

    def mcmc(self,x_test):
        xo = []
        xm = []
        x_test=self.data_standardization(x_test)
        for x in x_test:
            if 0 in x:
                xm.append(x)
            else:
                xo.append(x)
        xo = np.asarray(xo, dtype=np.float32)
        xm = np.asarray(xm, dtype=np.float32)
        xo_len = xo.shape[0]
        x_input = np.concatenate([xo, xm], 0)
        x_pre = self.Predicte(x_input)
        xm_pre = x_pre[xo_len:-1, :]
        output = np.concatenate([xo, xm_pre], 0)
        return output

    def data_standardization(self,dataset):
        for data in dataset:
            if "null" in data:
                for index, value in enumerate(data):
                    if value == "null":
                        data[index] = 0
        return np.asarray(dataset, dtype=np.float32)



