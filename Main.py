from vae_class import VAE
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


save_path = 'model/'
model_name = 'vae_model.ckpt'

def train_model(x):
    vae_model = VAE(x,h_dim=160,w_dim=49,z_dim=200,save_path=save_path,model_name=model_name,epoch=1000,bath_size=0)
    vae_model.Train()

def pre_model(x):
    vae_model = VAE(x, h_dim=160, w_dim=49, z_dim=200, save_path=save_path, model_name=model_name, epoch=1000,
                    bath_size=0)
    pre_X=vae_model.Predicte(x)
    return pre_X


def mcmc(x):
    vae_model = VAE(x, h_dim=160, w_dim=49, z_dim=200, save_path=save_path, model_name=model_name, epoch=1000,
                    bath_size=0)
    return vae_model.mcmc(x)

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

x = mnist.train.images[0:500]

x[100:200,:]=1
train_model(x)
pre_model(x)
mcmc(x)