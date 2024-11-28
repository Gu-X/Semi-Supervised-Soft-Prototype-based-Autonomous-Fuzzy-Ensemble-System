# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 15:27:45 2023

@author: XwGu
"""
import sklearn.metrics

def performancemeas_mc(y1,ye0): # performance measure for multi-class classification
    acc=sklearn.metrics.accuracy_score(y1,ye0)  # classification accuracy
    bacc=sklearn.metrics.balanced_accuracy_score(y1,ye0) # balanced classification accuracy
    f1=sklearn.metrics.f1_score(y1,ye0,average='weighted') # f1 scores
    mcc=sklearn.metrics.matthews_corrcoef(y1,ye0) # matthews correlation coefficient 
    return acc,bacc,f1,mcc

def performancemeas_bc(y1,ye0):
    acc=sklearn.metrics.accuracy_score(y1,ye0) # classification accuracy
    bacc=sklearn.metrics.balanced_accuracy_score(y1,ye0) # balanced classification accuracy
    f1=sklearn.metrics.f1_score(y1,ye0) # f1 scores
    mcc=sklearn.metrics.matthews_corrcoef(y1,ye0) # matthews correlation coefficient 
    return acc,bacc,f1,mcc