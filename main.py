# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 13:54:55 2023

@author: XwGu
"""


import numpy
from ASSLAFISEnsemble import ASSLAFISensmeble
from PerformanceMeasure import performancemeas_bc
import time
from matdataloader import matdataload

granularity=9
ensemblescale=10
splittingratio=0.9
numreceiver=2
confscorethreshold=0.9

CT=1
acc2a=numpy.zeros((CT,))
f12a=numpy.zeros((CT,))
bacc2a=numpy.zeros((CT,))
mcc2a=numpy.zeros((CT,))

acc3a=numpy.zeros((CT,))
f13a=numpy.zeros((CT,))
bacc3a=numpy.zeros((CT,))
mcc3a=numpy.zeros((CT,))
acc1a=numpy.zeros((CT,))
f11a=numpy.zeros((CT,))
bacc1a=numpy.zeros((CT,))
mcc1a=numpy.zeros((CT,))
    
count=0

NN=1

acc1=numpy.zeros((NN,))
f11=numpy.zeros((NN,))
bacc1=numpy.zeros((NN,))
mcc1=numpy.zeros((NN,))

acc2=numpy.zeros((NN,))
f12=numpy.zeros((NN,))
bacc2=numpy.zeros((NN,))
mcc2=numpy.zeros((NN,))

acc3=numpy.zeros((NN,))
f13=numpy.zeros((NN,))
bacc3=numpy.zeros((NN,))
mcc3=numpy.zeros((NN,))

ckratio=0.05;

for ii in range(0,NN):
    print(ii)
    dr='Data/CICIDS2017/PC1/data_'+str(ii+1) +'.mat'
    data0,y0,data10,y10,data1,y1=matdataload(dr)
    chunksize=10000#int((y0.shape[0]+y10.shape[0]+y1.shape[0])*ckratio)
    
    
    syst0=ASSLAFISensmeble(ensemblescale,granularity,chunksize,splittingratio,numreceiver,confscorethreshold)
    syst0.training(data0,y0) 
    ye0,score0=syst0.testing(data1)
    acc1[ii],bacc1[ii],f11[ii],mcc1[ii]=performancemeas_bc(y1,ye0)
    print([acc1[ii],bacc1[ii],f11[ii],mcc1[ii]])
    syst0.assltraining(data10,y10) 
    
    ye1,score1=syst0.testing(data1)
    acc2[ii],bacc2[ii],f12[ii],mcc2[ii]=performancemeas_bc(y1,ye1)
    print([acc2[ii],bacc2[ii],f12[ii],mcc2[ii]])

    start = time.time()
    seq1=[i for i in range(0,syst0.al_syst.NP) if numpy.max(syst0.al_syst.plabel[i,:])>=0.999]
    syst0.activelearning_prototypesfuision_manual(seq1,numpy.argmax(syst0.al_syst.plabel[seq1,:],axis=1)) 
    ye2,score2=syst0.testing(data1)
    acc3[ii],bacc3[ii],f13[ii],mcc3[ii]=performancemeas_bc(y1,ye2)
    print([acc3[ii],bacc3[ii],f13[ii],mcc3[ii]])

acc1a[count]=numpy.mean(acc1)
bacc1a[count]=numpy.mean(bacc1)
f11a[count]=numpy.mean(f11)
mcc1a[count]=numpy.mean(mcc1)

acc2a[count]=numpy.mean(acc2)
bacc2a[count]=(numpy.mean(bacc2))
f12a[count]=(numpy.mean(f12))
mcc2a[count]=(numpy.mean(mcc2))


acc3a[count]=(numpy.mean(acc3))
bacc3a[count]=(numpy.mean(bacc3))
f13a[count]=(numpy.mean(f13))
mcc3a[count]=(numpy.mean(mcc3))

Results1=numpy.array([acc1a,bacc1a,f11a,mcc1a])
Results2=numpy.array([acc2a,bacc2a,f12a,mcc2a])
Results3=numpy.array([acc3a,bacc3a,f13a,mcc3a])
print(Results1)
print(Results2)
print(Results3)
