# -*- coding: utf-8 -*-
"""
Created on Thu May 25 10:53:02 2023

@author: XwGu
"""
from AFISmodel import AFIS
from ALAFISmodel import ALAFIS
import numpy
import sklearn.metrics
import random
from sklearn.neighbors import KNeighborsClassifier
from Sampling import datasampling
import copy

class ASSLAFISensmeble:
    def __init__(self,ensemblescale,granularity,chunksize,splittingratio,numreceiver,confscorethreshold): # initialize the ensemble framework
        self.granularity=granularity # same level of granularity for AFIS
        self.splittingratio=splittingratio # the ratio for splitting the received labelled training data to the training and validation data pools
        self.numreceiver=numreceiver # given a particular input sample, the number of base classifiers that will receive it
        self.chunksize0=chunksize
        self.chunksize=int(chunksize/ensemblescale*self.numreceiver) # same chunk size for AFIS
        self.confscorethreshold=confscorethreshold # the level of confidence threshold needed for the ensemble system to confidently assign a pseudo label to an unlabelled sample
        self.confscorethreshold_automatedLabelling=0.99 # the level of confidence threshold needed for the ensemble system to confidently assign a pseudo label to a prototype learned from challenging unlabelled data
        self.maxtrainingpoolsize=int(self.chunksize*self.splittingratio)  # maximum size of the training data pool
        self.maxvalidationpoolsize=int(self.chunksize) # maximum size of the validation data pool
        self.maxultrainingpoolsize=int(self.chunksize) # maximum size of the pseudo labelled data pool
        self.trainingpoolx={} # initialize the collection of empty training data pools (data)
        self.validationpoolx={} # initialize the collection of empty validation data pools (data)
        self.trainingpooly={} # initialize the collection ofempty training data pools (corresponding label)
        self.validationpooly={} # initialize the collection of empty validation data pools (corresponding label)
        self.ultrainingpoolx={} # initialize the collection of empty pseudo labelled data pools (data)
        self.ultrainingpooly={} # initialize the collection of empty pseudo labelled data pools (corresponding label)
        self.syst={}  # store the paramaters of AFISs learned from data (both labelled and unlabelled)
        self.syst0={} # store the parameters of AFISs learned from labelled data and use them for initilizing AFISs with poor performances
        self.ast_syst={} # store the parameters of decision tree/nearest neighbour classifiers trained with labelled data and use them for pseudo labelling
        self.ast_syst_num={}  # number of DT/NN classifiers learned from labelled data per ensemble block
        self.ensemblescale=ensemblescale #  number of ensemble blocks in the ensemble system
        self.tolerance=1 # tolerance for weaker base classifier 
        self.pmeas=numpy.zeros((self.ensemblescale,)) # out-of-sample classification performance of each AFIS (in each ensemble block) measured by balanced accuracy
        self.pmeas0=self.pmeas # out-of-sample classification performance of each AFIS after training with labelled data
        self.ulNN=0
        for ii in range(0,self.ensemblescale): 
            self.syst[ii]=AFIS(self.granularity,self.chunksize) # initilize the AFIS templates for ensemble framework
            self.ast_syst[ii]={} # initilize the collection for storing the DT/NN classifiers in each ensemble block
            self.ast_syst_num[ii]=0 # the number of NN classifiers in each ensemble block
            self.trainingpoolx[ii]=numpy.array([]) # initialize the empty training data pool in each ensemble block (data)
            self.trainingpooly[ii]=numpy.array([]) # initialize the empty training data pool in each ensemble block (label)
            self.validationpoolx[ii]=numpy.array([]) # initialize the empty validation data pool in each ensemble block (data)
            self.validationpooly[ii]=numpy.array([]) # initialize the empty validation data pool in each ensemble block (label)
            self.ultrainingpoolx[ii]=numpy.array([]) # initialize the empty pseudo labelled data pool in each ensemble block (data)
            self.ultrainingpooly[ii]=numpy.array([]) # initialize the empty pseudo labelled pool in each ensemble block (label)
        self.al_syst=ALAFIS(self.chunksize,3) # initilize the active learning AFIS (ALAFIS) model to learn from highly challenging unlabelled data samples
        
    def training(self,data0,y0): # train the ensemble framework with labelled training data
    # data0: labelled training data
    # y0: the class labels of the data
        L0,W=data0.shape
        seq0=numpy.append(numpy.array(range(0,L0,self.chunksize0)),L0) # divide the recieved labelled training data into chunks (each ensemble block will recieve one portion of the data chunk with the size set by user)
        self.classes=numpy.unique(y0) # get the unique class labels of dara
        self.CL=self.classes.shape[0] # get the number of classes in the labelled training data
        for ii in range(0,seq0.shape[0]-1): 
            data00=data0[range(seq0[ii],seq0[ii+1]),:].copy() # read the current data chunk
            y00=y0[range(seq0[ii],seq0[ii+1])].copy() # read the labels of the current data chunk
            L=int(seq0[ii+1]-seq0[ii]) 
            seq=numpy.random.permutation(L)
            L1=int(L/self.ensemblescale*self.numreceiver)
            for kk in range(0,self.ensemblescale): # assign a portion of the data chunk to each base classifier
                seq1=numpy.random.choice(L,L1)
                seq2=seq[seq1].copy()
                data01=data00[seq2[0:int(L1*self.splittingratio)],:] # divide the current portion of data into training and validation samples according to the preset splitting ratio (training data)
                y01=y00[seq2[0:int(L1*self.splittingratio)]] # divide the current portion of data into training and validation samples according to the preset splitting ratio (labels of training data)
                data02=data00[seq2[int(L1*self.splittingratio):-1],:] # divide the current portion of data into training and validation samples according to the preset splitting ratio (validation data)
                y02=y00[seq2[int(L1*self.splittingratio):-1]] # divide the current portion of data into training and validation samples according to the preset splitting ratio (labels of validation data)
                if len(numpy.unique(y00[seq2]))>1: # if the current portion of the data chunk is composed of data samples of multiple classes
                    ## 
                    self.ast_syst[kk][self.ast_syst_num[kk]]=KNeighborsClassifier(n_neighbors=random.randint(1,15)) # create an empty NN template with a randomly generated "k" between 1 and 15 (number of nearest neighbours to be considered)
                    ##
                    self.ast_syst[kk][self.ast_syst_num[kk]]=self.ast_syst[kk][self.ast_syst_num[kk]].fit(data00[seq2,:],y00[seq2].reshape(len(seq2),)) # train the template with all the training data
                    self.ast_syst_num[kk]=self.ast_syst_num[kk]+1 # number of pseudo classifiers increased by 1
                if ii == 0 and self.validationpoolx[kk].size==0:  # initialize the training and validation data pools with the current portion of data
                    self.validationpoolx[kk]=data02
                    self.trainingpoolx[kk]=data01
                    self.validationpooly[kk]=y02
                    self.trainingpooly[kk]=y01
                else: # update the training and validation data pools with the current portion of data
                    self.validationpoolx[kk]=numpy.append(self.validationpoolx[kk],data02,axis=0)
                    self.validationpooly[kk]=numpy.append(self.validationpooly[kk],y02,axis=0)
                    self.trainingpoolx[kk]=numpy.append(self.trainingpoolx[kk],data01,axis=0)
                    self.trainingpooly[kk]=numpy.append(self.trainingpooly[kk],y01,axis=0)   
                    seq1=numpy.random.choice(self.trainingpoolx[kk].shape[0],self.maxtrainingpoolsize) # random sampling to restore the original size of training pool
                    seq2=numpy.random.choice(self.validationpoolx[kk].shape[0],self.maxvalidationpoolsize) # random sampling to restore the original size of validation pool
                    self.validationpoolx[kk]=self.validationpoolx[kk][seq2,:]
                    self.validationpooly[kk]=self.validationpooly[kk][seq2]
                    self.trainingpoolx[kk]=self.trainingpoolx[kk][seq1,:]
                    self.trainingpooly[kk]=self.trainingpooly[kk][seq1]
                data01,y01=datasampling.oversampling(data01,y01)  # perform oversampling to restore class balance by augmenting the amount fo minor class data samples
                self.syst[kk].training(data01,y01) # train the base classifier with the current data chunk
                ye0,score0,score1=self.syst[kk].testing(self.validationpoolx[kk]) # validate the out-of-sample classification performance of the trained AFIS
                bacc=sklearn.metrics.balanced_accuracy_score(self.validationpooly[kk],ye0) # measure the performance in terms of balanced accuracy
                self.pmeas[kk]=(self.pmeas[kk]+bacc)*0.5 # update the classifier weight of the base classifier according to all the processed historical data
            seq3=[i for i in range(0,self.ensemblescale) if self.pmeas[i]<numpy.mean(self.pmeas)-self.tolerance*numpy.std(self.pmeas)] # identify the weaker AFIS(s)
            if len(seq3)>0: # prune the weaker AFIS and replace it with a new one
                for jj in range(0,len(seq3)):
                    self.syst[seq3[jj]]=AFIS(self.granularity,self.chunksize) # create a new AFIS template to replace the weaker one
                    self.syst[seq3[jj]].training(self.trainingpoolx[seq3[jj]],self.trainingpooly[seq3[jj]]) # train the new AFIS with data stored in the corresponding training data pool
                    kk=numpy.random.choice(self.ensemblescale,1)[0]
                    while kk==seq3[jj]:
                        kk=numpy.random.choice(self.ensemblescale,1)[0]
                    self.syst[seq3[jj]].training(self.trainingpoolx[kk],self.trainingpooly[kk]) # train the new AFIS with data stored in  another randomly chosen training data pool
                    ye0,score0,score1=self.syst[seq3[jj]].testing(self.validationpoolx[seq3[jj]]) # evaluate the performance of the trained AFIS on validation samples
                    self.pmeas[seq3[jj]]=sklearn.metrics.balanced_accuracy_score(self.validationpooly[seq3[jj]],ye0) # measure the performance in terms of balanced accuracy
            for kk in range(0,self.ensemblescale):
                self.syst0[kk]=copy.deepcopy(self.syst[kk])  # store the parameters of AFISs trained with labelled training data and freeze them for later AFIS initilization
                self.pmeas0[kk]=self.pmeas[kk] # store the out-of-sample classification performances of AFISs trained with labelled training data and freeze them for later AFIS initilization

    def assltraining(self,data0,y0): # train the ensemble framework with unlabelled data in an active semi-supervised manner
    # data0: unlabelled training data
        labelavailblility=len(y0)!=0
        L0,W=data0.shape 
        seq0=numpy.append(numpy.array(range(0,L0,self.chunksize0)),L0) # divide the recieved unlabelled training data into chunks 
        self.ast_prob=numpy.zeros((L0,self.CL))
        for ii in range(0,seq0.shape[0]-1): 
            data11=data0[range(seq0[ii],seq0[ii+1]),:].copy() # read the current data chunk 
            plindx,plabels,ulindx=self.twostage_pseudolabeling(data11) # perform pseudo labelling on the current data chunk
            # plindx is the indices of unlabelled data samples that have been assigned with a pseudo label
            # plabels are corresponding pseudo labels assigned to the selected set of unlabelled data samples
            # ulindx is the indices of unlabelled data samples that cannot be pseudo labels (challenging samples)
            #print(plindx)
            plindx=list(map(int,plindx))
            ulindx=list(map(int,ulindx))
            if len(plindx)>0: 
                pldata=data11[plindx,:].copy() 
                L=plabels.shape[0]
                seq=numpy.random.permutation(L)
                L1=int(L/self.ensemblescale*self.numreceiver)
                for kk in range(0,self.ensemblescale): # assign data to each base classifier
                    seq1=numpy.random.choice(L,L1)
                    seq2=seq[seq1].copy()
                    data01=pldata[seq2,:] # randomly distribute pseudo labelled data samples to the AFIS in each ensemble block
                    y01=plabels[seq2]  # the corresponding pseudo labels of such data samples
                    if ii == 0 and self.ultrainingpooly[kk].size==0:  # initialize the pseudo labelled data pools with the current portion of pseudo labelled data
                        self.ultrainingpoolx[kk]=data01
                        self.ultrainingpooly[kk]=y01
                    else: # initialize the pseudo labelled data pools 
                        self.ultrainingpoolx[kk]=numpy.append(self.ultrainingpoolx[kk],data01,axis=0)
                        self.ultrainingpooly[kk]=numpy.append(self.ultrainingpooly[kk],y01,axis=0)   
                        seq1=numpy.random.choice(self.ultrainingpoolx[kk].shape[0],self.maxultrainingpoolsize)  # random sampling to restore the original size of the pseudo labelled data pool
                        self.ultrainingpoolx[kk]=self.ultrainingpoolx[kk][seq1,:]
                        self.ultrainingpooly[kk]=self.ultrainingpooly[kk][seq1]
                    data01,y01=datasampling.downsampling(data01,y01) # perform downsampling to restore class balance by reducing the amount fo major class data samples
                    self.syst[kk].training(data01,y01) # train the base classifier with the current portion of pseudo labelled data
                    ye0,score0,score1=self.syst[kk].testing(self.validationpoolx[kk]) # validate the performance of the semi-supervised AFIS
                    bacc=sklearn.metrics.balanced_accuracy_score(self.validationpooly[kk],ye0) # measure the performance in terms of balanced accuracy
                    self.pmeas[kk]=(self.pmeas[kk]+bacc)*0.5 # update the classifier weight of the semi-supervised AFIS
            ##
            self.ulNN=self.ulNN+len(ulindx)
            if len(ulindx)>0: # train the ALAFIS model with these challenging samples that are not assigned with pseudo labels
                averdist1=0
                for kk in range(0,self.ensemblescale):
                    averdist1=averdist1+self.syst[kk].averdist
                averdist1=averdist1/self.ensemblescale # get the average self-adaptive distance threshold used by ALAFIS
                if labelavailblility:
                    y11=y0[range(seq0[ii],seq0[ii+1]),:].copy() # read the current data chunk 
                    y11=y11[ulindx].copy()
                else:
                    y11=numpy.ones(len(ulindx),dtype=int)*self.CL 
                self.al_syst.training(data11[ulindx,:],y11.reshape(-1,),averdist1) # train the ALAFIS model
    
            seq3=[i for i in range(0,self.ensemblescale) if self.pmeas[i]<numpy.mean(self.pmeas)-self.tolerance*numpy.std(self.pmeas)] # identify the weaker AFIS models
            if len(seq3)>0: # prune the weaker AFIS and replace it with a new one
                for jj in range(0,len(seq3)):
                    self.syst[seq3[jj]]=copy.deepcopy(self.syst0[seq3[jj]]) # initialize the new AFIS template with the parameters learned from labelled data
                    self.syst[seq3[jj]].training(self.ultrainingpoolx[seq3[jj]],self.ultrainingpooly[seq3[jj]]) # train the AFIS model with the corresponding pseudo labelled data pool
                    kk=numpy.random.choice(self.ensemblescale,1)[0] # randomly choose another pseudo labelled data pool
                    while kk==seq3[jj]:
                        kk=numpy.random.choice(self.ensemblescale,1)[0]
                    self.syst[seq3[jj]].training(self.ultrainingpoolx[kk],self.ultrainingpooly[kk])  # train the AFIS model with the randomly selected pseudo labelled data pool
                    ye0,score0,score1=self.syst[seq3[jj]].testing(self.validationpoolx[seq3[jj]])  # evaluate the performance of the trained AFIS on validation samples
                    self.pmeas[seq3[jj]]=(self.pmeas0[seq3[jj]]+sklearn.metrics.balanced_accuracy_score(self.validationpooly[seq3[jj]],ye0))*0.5 # update the classifier weight of the AFIS model 
                
    def twostage_pseudolabeling(self,data00): # two stage pseduo labelling mechanism
    # data00: unlabelled data for pseudo labelling
        L=data00.shape[0]
        predictedlabels=numpy.zeros((L,self.ensemblescale)) 
        temp1=numpy.zeros((L,self.CL))
        for kk in range(0,self.ensemblescale):
            temp=numpy.zeros((L,self.CL))
            for jj in range(0,self.ast_syst_num[kk]):
                temp=temp+self.ast_syst[kk][jj].predict_proba(data00) # make predictions using the DT/NN classifier in each ensemble block
            predictedlabels[:,kk]=numpy.argmax(temp,axis=1) # get the predicted labels by each ensemble block using majority voting
            temp1=temp1+numpy.eye(self.CL)[predictedlabels[:,kk].astype(int)]
        predictedlabels0=numpy.argmax(temp1,axis=1) 
        plindex=[i for i in range(0,L) if len(numpy.unique(predictedlabels[i,:]))==1] # select out these unlabelled data samples with the predicted labels that all the pseudo-labelling ensembles agree (first stage)
        tempseq3=numpy.array(range(0,L))
        tempseq3=numpy.delete(tempseq3,plindex) # get the remaining unlabelled samples that are not assigned a pseudo label at the first stage of the pseudo labelling process
        ulindex=numpy.array(range(0,0)) # create a challenging sample list
        pseudolabels=predictedlabels0[plindex] # get the pseudo labels for the data samples selected out during the first stage
        if len(tempseq3)>0:   
            y22,score00=self.testing(data00[tempseq3,:].copy()) # use the AFIS models within the ensemble framework to make predictions on their class labels
            L1=len(tempseq3)
            predictedlabels1=predictedlabels0[tempseq3].copy()       
            tempseq4=[i for i in range(0,L1) if numpy.max(score00[i,:])>self.confscorethreshold and predictedlabels1[i]==y22[i]] # get these data samples with the predicted class labels that the ensemble framework has very high confidence (second stage)
            ulindex=numpy.array(range(0,L1)) 
            ulindex=numpy.delete(ulindex,tempseq4) # remove these data samples with high confident predicted labels from the challenging sample list
            ulindex=tempseq3[ulindex].copy() 
            plindex=numpy.append(plindex.copy(),tempseq3[tempseq4].copy(),axis=0) # expand the list of data samples assigned with pseudo labels
            pseudolabels=numpy.append(pseudolabels.copy(),y22[tempseq4],axis=0) # expand the collection of pseudo labels
        return plindex,pseudolabels,ulindex
                 
    def activelearning_prototypesfuision_auto(self,data0,y0): # automated labelling for prototypes learned actively from challenging unlabelled samples
    # data0: labelled training data (can be the same data used for priming the ensemble model)
    # y0: labels of data0
        neigh = KNeighborsClassifier(n_neighbors=3) 
        neigh.fit(data0,y0) # train the KNN model with k=3
        proba=neigh.predict_proba(self.al_syst.ulprototype) # find three nearest data smaples to each of the unlabeled challenging prototype
        seq1=[i for i in range(0,self.al_syst.NP) if max(proba[i,:])==1]  # find these challenging prototypes whose nearest data samples all belong to the same class
        self.activelearning_prototypesfuision_manual(seq1,numpy.argmax(proba[seq1,:],axis=1)) # assign class labels to these challenging prototypes whose nearest data samples all belong to the same class

        
    def activelearning_prototypesfuision_manual(self,idx,labels): # manual labelling for prototypes learned actively from challenging unlabelled samples
        NC1=len(idx)
        prototypes1=self.al_syst.ulprototype[idx,:]
        support1=self.al_syst.support[idx]
        self.al_syst.databaseclearning(idx) # remove these selected prototypes from the ALAFIS model
        self.alprototypeintegration(prototypes1,NC1,support1,labels) # manual labelling for prototypes learned actively from challenging unlabelled samples
        
    def alprototypeintegration(self,prototypes1,NC1,support1,labels): # merge the labelled prototypes into the knowledge bases of the AFIS models
        for jj in range(0,self.ensemblescale):
            for kk in range(0,self.syst[jj].CL):
                rnseq=numpy.random.rand(NC1,)
                seq=[i for i in range(0,NC1) if labels[i]==kk and rnseq[i]>=0.5]
                self.syst[jj].prototypeintegration(prototypes1[seq,:],len(seq),support1[seq],kk)

    def testing(self,data1):  # use the ensemble framework to make predictions on the unlabelled data
    # data1: unlabelled data for testing
        score0={}
        score1={}
        ye0={}
        seq3=[i for i in range(0,self.ensemblescale) if self.pmeas[i]<numpy.mean(self.pmeas)-self.tolerance*numpy.std(self.pmeas)]
        self.pmeas[seq3]=0 # exclude these weaker AFIS models from joint decison making
        score00=numpy.zeros((data1.shape[0],self.CL))
        for kk in range(0,self.ensemblescale):
            if self.pmeas[kk]!=0:
                ye0[kk],score0[kk],score1[kk]=self.syst[kk].testing(data1) # produce the confidence scores on each data sample
            else:
                score0[kk]=numpy.zeros((data1.shape[0],self.CL))
            score00=score00+score0[kk]*self.pmeas[kk] # perform weighted joint decison-making
        score00=score00/sum(self.pmeas)
        return numpy.argmax(score00,axis=1),score00 # predicted class labels and the confidence scores
    
    
    def testing_visualization(self,data1):  # use the ensemble framework to make predictions on the unlabelled data (this function is used for visualizing the inferencing process)
    # data1: unlabelled data for testing
        score0={}
        NNPindex={} 
        ye0={}
        attrdiff={}
        seq3=[i for i in range(0,self.ensemblescale) if self.pmeas[i]<numpy.mean(self.pmeas)-self.tolerance*numpy.std(self.pmeas)]
        self.pmeas[seq3]=0 # exclude these weaker AFIS models from joint decison making
        score00=numpy.zeros((data1.shape[0],self.CL))
        for kk in range(0,self.ensemblescale):
            if self.pmeas[kk]!=0:
                ye0[kk],score0[kk],NNPindex[kk],attrdiff[kk]=self.syst[kk].inferencing_visualization(data1) # the kk-th AFIS model produces the corresponding class labels, confidence scores and the indices of the "nnp" nearest prototypes for unlabelled testing data samples
            else:
                score0[kk]=numpy.zeros((data1.shape[0],self.CL))
            score00=score00+score0[kk]*self.pmeas[kk] # perform weighted joint decison-making 
        score00=score00/sum(self.pmeas)
        return numpy.argmax(score00,axis=1),score00,NNPindex,attrdiff # predicted class labels,  the confidence scores and the indices of the "nnp" nearest prototypes of all the AFIS models in the ensemble framework
    
