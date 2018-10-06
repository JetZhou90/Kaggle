from keras.applications import VGG19
from keras.layers import Dense,Activation,Flatten,Multiply,Embedding
from keras.models import  Sequential, Model
import numpy as np
import tensorflow as tf
from keras import backend as K
from PIL import Image
import cv2
import matplotlib.pyplot as plt


class ZSL_model:
    def __init__(self,learning_rate=1e-8,batch_size=128):
        self.LEARNING_RATE=learning_rate
        self.BATH_SIZE=batch_size


    def build_model(self,images):
        numImage,h,w,c=images.shape
        # 1st Scale F_NET to gain the image features
        # conv_feature gained from last conv_layer
        # image_feature gained from last ful-con layer

        vgg19= VGG19(include_top=False,weights='vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
                     input_shape=(h,w,c))
        for layer in vgg19.layers:
            layer.trainable = False
        F_net=Model(vgg19.input,vgg19.layers[-2].output)
        # F_net.summary()
        conv_features = F_net.output
        # conv_features=F_net.layers[-2].output
        # print(np.shape(conv_features))
        # Zoom Net
        conv_features = Flatten()(conv_features)
        zoom_feature  = Dense(1024)(conv_features)
        zoom_output= Dense(3)(zoom_feature)
        zoom_output=Activation('sigmoid')(zoom_output)
        # Zoom in the image
        # f(x) = 1/(1 + exp(-kx)) and k is set to 10
        # m_x=f(x - z_x + 0.5*z_s) - f(x-z_x-0.5*z_s)
        # m_y=f(y - z_y + 0.5*z_s) - f(y-z_y-0.5*z_s)
        new_images = Multiply()([F_net.input, zoom_output])
        z_net = Model(inputs=F_net.input, outputs=new_images)
        # print(new_images)
        # loss=mean_squared_error(z_net.input,new_images)
        z_net.compile(optimizer='adam',loss='mse')
        # z_net.summary()
        return z_net

    def build_embedding(self,images,k):
        numImage, h, w, c = images.shape
        # 1st Scale F_NET to gain the image features
        # conv_feature gained from last conv_layer
        # image_feature gained from last ful-con layer

        vgg19 = VGG19(include_top=False, weights='vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
                      input_shape=(h, w, c))
        conv_features = vgg19.layers[-2].output
        image_feature=Flatten()(conv_features)
        image_feature=Dense(1024)(image_feature)
        image_feature=Dense(256)(image_feature)
        image_feature=Activation('relu')(image_feature)
        F_net2 = Model(vgg19.input, image_feature)
        # F_net2.summary()
        embedding_output = Dense(2*k)(F_net2.output)
        at_x = Activation('softmax')(embedding_output[:, 0:k])
        lat_x =Activation('sigmoid')( embedding_output[:, k:-1])
        softloss=K.mean(K.sum(-K.log(K.max(at_x))))

        E_net = Model(F_net2.input,embedding_output)
        E_net.summary()

        return E_net




        # conv_features=F_net.layers[-2].output
        # print(np.shape(conv_features))
        # Zoom Net
        # conv_features = Flatten()(conv_features)




image=cv2.imread('test.jpg')
image=cv2.resize(image,(224,224))
# image=Image.open('test.JPG')
# image=image.resize(224,224)
image=np.asarray(image,dtype=np.float32)/255.

images=np.asarray([image,image,image,image,image,image],dtype=np.float32)

# z_model=ZSL_model().build_model(images)
# z_model.fit(images,images,epochs=3,batch_size=1)
# # new_images=cv2.resize(new_images,(224,224),interpolation=cv2.INTER_LINEAR)
# new_images=z_model.predict(images)
# new_image=new_images[0]
# print(np.mean(image-new_image))
# new_image=cv2.resize(new_image,(224,224),dst=image,interpolation=cv2.INTER_LINEAR)
# cv2.imshow('image',image)
# cv2.imshow('new_image',new_image)
# cv2.waitKey()
# cv2.destroyAllWindows()
k=10
e_model=ZSL_model().build_embedding(images,k)
# at_x  = e_model.output[:,0:k]
# lat_x = e_model.output[:,k:-1]






