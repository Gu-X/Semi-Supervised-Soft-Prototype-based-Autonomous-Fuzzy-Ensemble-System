# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 11:11:04 2023

@author: XwGu
"""

import numpy
import scipy
class ALAFIS: # AFIS model for active learning
    def __init__(self,chunksize,pnclass): # create an empty AFIS template for active learning
        self.pnclass=pnclass # the artificial class of these challenging data samples
        self.chunksize=int(chunksize) # the chunk size to learn from data streams
        self.chunksize1=int(chunksize*5) # the internal chunk size to avoid exceeding the system memory limitation
        self.p1=1 # parameter_1 for calculating Minkowski distance
        self.p2=2 # parameter_2 for calculating Minkowski distance
        self.ulprototype={} # empty set of soft prototypes (to be learned from data)
        self.support={} # empty set of supports (number of parameters associated with soft prototyes,to be learned from data)
        self.NP=0 # number of soft prototypes per class
        self.CL=0 # number of classes (to be learned from data)
        self.ND=0 # number of data samples processed
        self.nnp=9 # number of nearest soft prototypes used for inference
        self.NDC=0 # number of data chunks processed
        self.plabel={} # the class labels of the prototypes used for distinguishing them from soft prototypes with class labels, which do not carry any meaning
        
    def training(self,data0,y0,averdist1):  # train the ALAFIS with the challenging data
    # data0: input data
    # y0: class labels of these data samples (default value is pnclass-1, but can be any value below pnclass-1)
    # averdist1: the distance threshold
        self.averdist=averdist1 
        L0,W=data0.shape
        seq0=numpy.append(numpy.array(range(0,L0,self.chunksize)),L0)
        for ii in range(0,seq0.shape[0]-1):
            self.NDC=self.NDC+1
            data00=data0[range(seq0[ii],seq0[ii+1]),:].copy()
            y00=y0[range(seq0[ii],seq0[ii+1])].copy()
            NL,W=data00.shape
            tempdist=self.pdist(data00)
            tempdist=scipy.spatial.distance.squareform(tempdist)
            NC,center,sup,clabel=self.prototypeidentification(data00,y00,tempdist) # identify soft prototypes from the current data chunk
            if self.NP==0:
                self.ulprototype=center.copy()
                self.support=sup.copy()
                self.NP=NC
                self.plabel=clabel.copy()
            else:
                self.prototypeintegration(center,NC,sup,clabel)  # merge the latest soft prototypes to the knowledge base 
                
    def databaseclearning(self,idx):  #  remove certain  prototypes from the data base 
        self.ulprototype=numpy.delete(self.ulprototype,idx,axis=0)
        self.support=numpy.delete(self.support,idx)
        self.plabel=numpy.delete(self.plabel,idx,axis=0)
        self.NP=self.NP-len(idx)
          
  #####  
  
    def prototypeidentification(self,data0,y0,dist0): # identify soft prototypes from the current input data (same as AFIS)
        data0,idx0=numpy.unique(data0,return_index=True,axis=0)
        y0=y0[idx0]
        oh_y0 =numpy.eye(self.pnclass)[y0]
        L,W=data0.shape
        if L>=3:
            dist0=dist0[idx0,:].copy()
            dist00=dist0[:,idx0]
            tempdist=numpy.zeros((L,L))
            tempdist[dist00<=self.averdist]=1
            dist1=numpy.exp(-1*dist00/self.averdist)
            tempseq=numpy.sum(tempdist*dist1.copy(),axis=1)
            tempdist=tempdist*tempseq
            tempseq=numpy.max(tempdist.copy(),axis=1)
            tempseq1=numpy.diag(tempdist.copy())
            tempseq2=numpy.array([i for i in range(L) if (tempseq1[i]-tempseq[i])==0])
            NC=tempseq2.shape[0]
            center=numpy.empty((NC,W))
            clabels=numpy.empty((NC,self.pnclass))
            sup=numpy.empty((NC,))
            dist2=numpy.zeros((NC,L))
            for ii in range(0,NC):
                 dist2[ii,:]=dist1[tempseq2[ii],:]/numpy.sum(dist1[tempseq2[ii],:])
                 center[ii,:]=numpy.sum((data0.transpose()*dist2[ii,:]),axis=1)
                 clabels[ii,:]=numpy.sum((oh_y0.transpose()*dist2[ii,:]),axis=1)
            sup=numpy.sum(dist2/numpy.sum(dist2,axis=0),axis=1)
        else:
            NC=L
            center=data0.reshape((L,W))
            sup=numpy.ones((L,))
            clabels=oh_y0
        return NC,center,sup,clabels
   
    def prototypeintegration(self,center1,NC1,sup1,clabel1): # merge the latestly identified soft prototyoes into the existing knowledge base (same as AFIS)
        seq1=numpy.append(numpy.array(range(0,self.NP,self.chunksize1)),self.NP)
        mdist2=numpy.full((NC1,), numpy.inf)
        dist0=numpy.zeros((self.NP,NC1),dtype=float)
        for jj in range(0,seq1.shape[0]-1):
            dist2=self.cdist(self.ulprototype[range(seq1[jj],seq1[jj+1]),:],center1)
            dist0[seq1[jj]:seq1[jj+1],:]=dist2.copy()
        mdist2=numpy.min(dist0,axis=0)
        seq3=[i for i in range(NC1) if mdist2[i]>self.averdist]
        seq4=[i for i in range(NC1) if mdist2[i]<=self.averdist]
        self.ulprototype=numpy.append(self.ulprototype,center1[seq3,:],axis=0)
        self.support=numpy.append(self.support,sup1[seq3],axis=0)
        self.plabel=numpy.append(self.plabel,clabel1[seq3,:],axis=0)
        dist3=self.cdist(center1[seq3,:],center1[seq4,:])
        self.NP+=sup1[seq3].shape[0]
        center1=center1[seq4,:].copy()
        sup1=sup1[seq4]
        clabel1=clabel1[seq4,:].copy()
        dist0=numpy.exp(-1*numpy.append(dist0[:,seq4].copy(),dist3,axis=0)/self.averdist)
        dist10=numpy.sum(dist0.copy(),axis=0)/sup1
        dist0=dist0.copy()/dist10
        temps1=numpy.sum(dist0,axis=1)
        self.ulprototype=(((self.ulprototype.transpose()*self.support)+numpy.matmul(dist0,center1).transpose())/(self.support+temps1)).transpose()
        self.plabel=(((self.plabel.transpose()*self.support)+numpy.matmul(dist0,clabel1).transpose())/(self.support+temps1)).transpose()
        self.support+=temps1
    
    def pdist(self,data0): # calculate the distances between data samples within the same set
        temp=scipy.spatial.distance.pdist(data0,'minkowski', p=self.p1)**self.p2/data0.shape[1]
        return temp
    
    def cdist(self,data0,data1): # calculate the pairwise distances between two sets of data samples
        temp=scipy.spatial.distance.cdist(data0,data1,'minkowski', p=self.p1)**self.p2/data0.shape[1]
        return temp    