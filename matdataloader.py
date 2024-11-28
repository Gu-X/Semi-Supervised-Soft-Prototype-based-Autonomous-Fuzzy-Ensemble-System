# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 14:19:35 2023

@author: XwGu
"""
import numpy
import scipy.io

def matdataload(dr):
    mat_contents = scipy.io.loadmat(dr)

    L1,W1=mat_contents['DTra1'].shape
    y0=numpy.zeros((L1,1),dtype='int')
    data0=numpy.zeros((L1,W1))
    for i in range(0,L1):
        y0[i,:]=mat_contents['LTra1'][i]-1
        data0[i,:]=mat_contents['DTra1'][i]
    
    L10,W10=mat_contents['DTra2'].shape
    y10=numpy.zeros((L10,1),dtype='int')
    data10=numpy.zeros((L10,W10))
    for i in range(0,L10):
        y10[i,:]=mat_contents['LTra2'][i]-1
        data10[i,:]=mat_contents['DTra2'][i]
        
    L2,W2=mat_contents['DTes1'].shape
    y1=numpy.zeros((L2,1),dtype='int')
    data1=numpy.zeros((L2,W2))
    for i in range(0,L2):
        y1[i,:]=mat_contents['LTes1'][i]-1
        data1[i,:]=mat_contents['DTes1'][i]
        
    
    return data0,y0,data10,y10,data1,y1


def matdataload_transductive(dr):
    mat_contents = scipy.io.loadmat(dr)

    L1,W1=mat_contents['DTra1'].shape
    y0=numpy.zeros((L1,1),dtype='int')
    data0=numpy.zeros((L1,W1))
    for i in range(0,L1):
        y0[i,:]=mat_contents['LTra1'][i]-1
        data0[i,:]=mat_contents['DTra1'][i]
        
    L2,W2=mat_contents['DTes1'].shape
    y1=numpy.zeros((L2,1),dtype='int')
    data1=numpy.zeros((L2,W2))
    for i in range(0,L2):
        y1[i,:]=mat_contents['LTes1'][i]-1
        data1[i,:]=mat_contents['DTes1'][i]
        
    
    return data0,y0,data1,y1