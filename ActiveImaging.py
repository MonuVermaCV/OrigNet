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


import cv2
import numpy as np 
import os

#path to dataset folder
path1="H:\\DEEP_LBP\\Micro_Original Dataset\\SAMM_Organized_Crop"
emo_folders = os.listdir(path1)


for folders in emo_folders:
    s_folder=os.listdir(path1+'\\'+folders)
    for sfolders in s_folder:
        sub_folder=os.listdir(path1+'\\'+folders+'\\'+sfolders)
        for subfolders in sub_folder:
            images=os.listdir(path1+'\\'+folders+'\\'+sfolders+'\\'+subfolders)
            #128x128=16384 size of image 
            imageArray=np.empty((len(images),16384,1))
            imageArray2=np.empty((len(images),16384,1))
            imageArray3=np.empty((16384,1))
            j=0
            for img in images:  
                
                im = cv2.imread(path1+'\\'+folders+'\\'+sfolders+'\\'+subfolders+'\\'+img)
                im1=cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY),(128,128))
                array=np.asarray(im1) 
                #128x128 size of image 
                arr=np.reshape(array,(128*128,1))
                imageArray[j]=arr
                j=j+1

            imageArray2[0]=np.zeros((16384,1),dtype=int)
            for j in range(1,len(images)): 
                imageArray3=imageArray2[0]
                for k in range(0,j-1):
                    imageArray3=imageArray3+(imageArray[j]-imageArray[k])
                imageArray2[j]=imageArray3

            res=np.sum(imageArray2,axis=0)
       
            w=np.reshape(res,(128,128))
            #destination path for active images
            imageName='H:\\DEEP_LBP\\Micro_Original Dataset\\SAMM_Organized_Crop - Copy'+'\\'+str(folders)+'\\'+str(sfolders)+'\\'+'\\image_'+str(subfolders)+'.jpg'
           
            
            cv2.imwrite(imageName,w)
        
    
    


