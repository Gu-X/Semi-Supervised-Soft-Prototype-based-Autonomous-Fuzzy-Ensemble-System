# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 22:35:59 2023

@author: XwGu
"""

import numpy
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler



class datasampling(): 
    def downsampling(data,y): # Perform random downsampling to restore the class imbalance
    # input:
    # 1) data: data samples for downsampling
    # 2) y:   corresponding class labels
        if len(numpy.unique(y))>1:
            # Perform random sampling to reduce the amount of data samples of the major class; data samples of minor class will be remain the same
            cc = RandomUnderSampler() # Create a random downsampling object
            data,y = cc.fit_resample(data,y) # Resample the data using the object 
        else:
            # if the data is composed of a single class (major), randomly remove 95% of the data
            L,W=data.shape
            seq=numpy.random.permutation(L)
            seq1=seq[int(L*0.05):L].copy()
            data=numpy.delete(data,seq1,axis=0)
            y=numpy.delete(y,seq1,axis=0)
        return data,y
    # output:
    # 1) data: data samples after downsampling
    # 2) y:   corresponding class labels
    
    def oversampling(data,y): # Perform oversampling using the Synthetic Minority Over-sampling Technique (SMOTE) algorithm to restore the class imbalance
    # input:
    # 1) data: data samples for oversampling
    # 2) y:   corresponding class labels
    # Perform SMOTE to augment the amount of data samples of the minor class; data samples of major class will be remain the same
        uy=numpy.unique(y)
        Ly=len(uy)
        LCy=numpy.zeros((Ly,),dtype=int)
        for qq in range(0,Ly):
            LCy[qq]=sum(y==uy[qq]) # get the number of data samples of each class
        sm =SMOTE(k_neighbors=numpy.min([5,numpy.min(LCy)])) # create the SMOTE object but only changing the default setting if the number of minority data samples is smaller than 5
        data,y=sm.fit_resample(data,y) # Resample the data using the object 
        return data,y
    # output:
    # data: data samples after oversampling
    # y:   corresponding class labels