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

from keras.layers import LeakyReLU,multiply,add,Activation



def RReLU(x):
        sig1 =Activation('sigmoid')(x)
        relu1=LeakyReLU(alpha=0.1)(x)
        mul=multiply([sig1,relu1])
        RReLU=add([sig1,mul])
        return RReLU