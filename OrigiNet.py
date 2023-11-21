#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
%This is an code for CNN architecture used in Non-Linearities Improve OrigiNet based on Active Imaging for Micro Expression Recognition
%{@autors: Verma, M., Vipparthi, S.K. and Singh, G., 2020. 
%Title: Non-Linearities Improve OrigiNet based on Active Imaging for Micro Expression Recognition
%This code is written by Monu Verma @ CSE, MNIT, Jaipur (INDIA)
%For any question you can contact us at monuverma.cv@gmail.com
%For more details you can also visit us: https://visionintelligence.github.io/

%----specify size of input-------

@author: Monu Verma
"""
from keras.models import Model
from keras.layers import Activation, LeakyReLU,multiply, Input,concatenate,Flatten,Dense,add,BatchNormalization
from keras.layers.convolutional import Conv2D


def RReLU(x):
        sig1 =Activation('sigmoid')(x)
        relu1=LeakyReLU(alpha=0.1)(x)
        mul=multiply([sig1,relu1])
        RReLU=add([sig1,mul])
        return RReLU
    
def build():
        im1 =Input(shape=(128,128,3))
        x1=Conv2D(16, (3,3),strides=2, padding='same', name='conv1_1')(im1)
        x1=RReLU(x1)
        y1=Conv2D(16, (5,5),strides=2, padding='same', name='conv1_2')(im1)
        y1=RReLU(y1)
        add1=add([x1,y1])
        BN1=BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None)(add1)
        x2=Conv2D(32, (3,3),strides=2, padding='same', name='conv2_1')(BN1)
        x2=RReLU(x2)
        y2=Conv2D(32, (5,5),strides=2, padding='same', name='conv2_2')(BN1)
        y2=RReLU(y2)
        add2=add([x2,y2])
        BN2=BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None)(add2)
        x3=Conv2D(64, (3,3),strides=2, padding='same', name='conv3_1')(BN2)
        x3=RReLU(x3)
        y3=Conv2D(64, (5,5),strides=2, padding='same', name='conv3_2')(BN2)
        y3=RReLU(y3)
        add3=add([x3,y3])
        BN3=BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None)(add3)
        x4=Conv2D(96, (3,3),strides=2, padding='same', name='conv4_1')(BN3)
        x4=RReLU(x4)
        x4=Flatten()(x4)
        x4=Dense(128)(x4)
        y4=Conv2D(96, (5,5),strides=2, padding='same', name='conv4_2')(add3)
        y4=RReLU(y4)
        y4=Flatten()(y4)
        y4=Dense(128)(y4)
        concat=concatenate([x4,y4])
        out = Dense(8, activation='softmax')(concat)
        model = Model(inputs=[im1],outputs= out)
        return model