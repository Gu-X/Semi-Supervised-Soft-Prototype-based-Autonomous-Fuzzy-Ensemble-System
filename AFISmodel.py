import numpy
import scipy

class AFIS:
    def __init__(self,granularity,chunksize): ## create an empty AFIS template
        self.granularity=int(granularity) # the level of granularity for prototype identification 
        self.chunksize=int(chunksize) # the chunk size to learn from data streams
        self.chunksize1=int(chunksize*5) # the internal chunk size to avoid exceeding the system memory limitation
        self.p1=1 # parameter_1 for calculating Minkowski distance
        self.p2=2 # parameter_2 for calculating Minkowski distance
        self.prototype={} # empty set of soft prototypes (to be learned from data)
        self.support={} # empty set of supports (number of parameters associated with soft prototyes,to be learned from data)
        self.NP={} # number of soft prototypes per class
        self.averdist=0 # self-adaptive distance threshold (to be learned from data)
        self.CL=0 # number of classes (to be learned from data)
        self.ND=0 # number of data samples processed
        self.nnp=9 # number of nearest soft prototypes used for inference
        self.NDC=0 # number of data chunks processed
        self.delta=0.001 # the ratio control the tolerance towards the similarity between prototypes in the knowledge base during prototype fusion
        self.fusIn=25 # the interval bewteen two successive prototype pruning operations (in terms of the number of data chunks being processed)
        
    def training(self,data0,y0):  # train the AFIS with the data in a supervised manner
    # data0: labelled training data
    # y0: the class labels of the data
        L0,W=data0.shape
        seq0=numpy.append(numpy.array(range(0,L0,self.chunksize)),L0)
        if self.ND==0:
            self.classes=numpy.unique(y0)
            self.CL=self.classes.shape[0]
        for ii in range(0,seq0.shape[0]-1):
            self.NDC=self.NDC+1
            data00=data0[range(seq0[ii],seq0[ii+1]),:].copy()
            y00=y0[range(seq0[ii],seq0[ii+1])].copy()
            NL,W=data00.shape
            tempdist=self.pdist(data00)
            temp0=self.thresholdestimation(tempdist) # estimate the local distance threshold
            tempdist=scipy.spatial.distance.squareform(tempdist) # get the distance matrix
            if self.ND==0:
                self.averdist=temp0 # initialize the self-adaptive distance threshold
                self.ND=NL # initialize number of processed data samples
            else:
                self.averdist=(self.averdist*self.ND+temp0*NL)/(self.ND+NL) # update the self-adaptive distance threshold
                self.ND=self.ND+NL # update number of processed data samples
            for kk in range(0,self.CL):
                tempseq=y00.reshape(NL,)==self.classes[kk]
                if data00[tempseq,:].shape[0]>0:
                    tempdist2=tempdist[tempseq,:].copy()
                    tempdist2=tempdist2[:,tempseq]
                    NC,center,sup=self.prototypeidentification(data00[tempseq,:],y00[tempseq],tempdist2) # identify soft prototypes from the current data chunk
                    if self.NP.get(kk) is None: # if there has been no prototypes of this particular class identified before
                        self.prototype[kk]=center.copy() # store the prototypes of the kk th class in the knowledge base
                        self.support[kk]=sup.copy() #  store the supports of prototypes of the kk th class in the knowledge base
                        self.NP[kk]=NC # store the number of prototypes of the kk th class  in the knowledge base
                    else:
                        self.prototypeintegration(center,NC,sup,kk)  # merge the latest soft prototypes to the knowledge base
            if self.NDC%self.fusIn==0 and self.NDC>0:
                for kk in range(0,self.CL):
                    if self.NP.get(kk) is not None:
                        self.prototypepruning(kk) # merge the highly similar soft prototypes together to reduce the size of knowledge base (periodically)
 ###                   
    def testing(self,data1): # use the trained AFIS to make predictions on unlabelled data 
    # data1: unlabelled data for testing
        L,W=data1.shape
        score0=numpy.zeros((L,self.CL))
        score1=numpy.zeros((L,self.CL))
        for kk in range(0,self.CL):
            score0[:,kk]=self.testingperclass(data1,L,kk) # calculate the confidence score on each data sample per class based on the similarity between it and prototypes
        score0_norm=numpy.sum(score0,axis=1)
        score0_norm[score0_norm==0]=1
        for kk in range(0,self.CL):
            score1[:,kk]=numpy.divide(score0[:,kk].copy(),score0_norm) # normalize the confidence scores of different classes so that they sum up to 1
        return numpy.argmax(score0,axis=1),score1,numpy.max(score0,axis=1)  
    # the first output: predicted class labels,
    # the second output: normalized confidence scores
    # the third output: highest confidence score given by AFIS
    
    def testingperclass(self,data1,L,kk): # make predictions on unlabelled data using the soft prototypes identified from data of the kth class
        seq0=numpy.append(numpy.array(range(0,L,self.chunksize)),L)
        seq1=numpy.append(numpy.array(range(0,self.NP[kk],self.chunksize1)),self.NP[kk])
        score0=numpy.zeros((L,))
        for ii in range(0,seq0.shape[0]-1):
            tempscore3=numpy.zeros((seq0[ii+1]-seq0[ii],self.nnp))
            tempdata=data1[range(seq0[ii],seq0[ii+1]),:].copy()
            for jj in range(0,seq1.shape[0]-1):
                dist1=self.cdist(tempdata,self.prototype[kk][range(seq1[jj],seq1[jj+1]),:]) # calculate the distances between prototypes and testing samples
                dist10=numpy.exp(-1*dist1.copy()/self.averdist)*self.support[kk][range(seq1[jj],seq1[jj+1])] # convert the distances to similarity scores
                tempscore3=numpy.append(tempscore3,dist10,axis=1)
                tempscore3=-numpy.sort(-tempscore3.copy(),axis=1)  
                tempscore3=tempscore3[:,0:self.nnp] # get the similarity scores of the "nnp" nearest prototypes 
            score0[range(seq0[ii],seq0[ii+1])]=numpy.sum(tempscore3,axis=1) # get the confidence scores as the average of the similarity scores produced by the "nnp" nearest prototypes
        return score0 # produce the confidence scores per class
    
    def inferencing_visualization(self,data1): # this function is used for visualizing the prediction process of the trained AFIS on unlabelled data
        L,W=data1.shape
        index_nnp={}
        score0=numpy.zeros((L,self.CL))
        for kk in range(0,self.CL):
            index_nnp[kk]=numpy.zeros((L,self.nnp),dtype=int)
        for ii in range(0,L):
            for kk in range(0,self.CL):
                dist1=self.cdist(data1[ii,:].reshape(1,W),self.prototype[kk]) # calculate the distances between prototypes and each testing sample
                dist10=numpy.exp(-1*dist1.copy()/self.averdist)*self.support[kk] # convert the distances to similarity scores
                tempindex=numpy.argsort(-1*dist10.copy(),axis=1) # get the indices of the "nnp" nearest prototypes that give the highest simlarity scores
                index_nnp[kk][ii,:]=tempindex[0,0:self.nnp] 
                score0[ii,kk]=numpy.sum(dist10[0,tempindex[0:self.nnp]],axis=1) # calculate the confidence score based on the similarity scores given by the "nnp" nearest prototypes
        score0_norm=numpy.sum(score0,axis=1)
        score0_norm[score0_norm==0]=1
        score1=numpy.zeros((L,self.CL))
        for kk in range(0,self.CL):
            score1[:,kk]=numpy.divide(score0[:,kk].copy(),score0_norm) # normalize the confidence scores such that they sum up to 1
        return numpy.argmax(score0,axis=1),score1,index_nnp
    # numpy.argmax(score0,axis=1): the predicted class labels of unlabelled data samples
    # score1: the scores of confidence produced by the trained AFIS on unlabelled data samples
    # index_nnp: each row gives the indices of the "nnp" nearest prototypes of each class for the corresponding unlabelled data sample 
  #####

    def prototypeidentification(self,data0,y0,dist0): # identify soft prototypes from the current input data
    # data0: the training samples
    # y0: class labels of the training samples
    # dist0: distance matrix 
        data0,idx0=numpy.unique(data0,return_index=True,axis=0) # remove the redundant training samples (repetition)
        y0=y0[idx0]
        L,W=data0.shape
        if L>=3:
            dist0=dist0[idx0,:].copy()
            dist00=dist0[:,idx0] # remove the columns and rows of redundant training samples
            tempdist=numpy.zeros((L,L))
            tempdist[dist00<=self.averdist]=1 # convert the distance matrix into an affinity matrix
            dist1=numpy.exp(-1*dist00/self.averdist) # convert the distance matrix into a similarity matrix
            tempseq=numpy.sum(tempdist*dist1.copy(),axis=1) # calculate the data density of each data sample
            tempdist=tempdist*tempseq
            tempseq=numpy.max(tempdist.copy(),axis=1)
            tempseq1=numpy.diag(tempdist.copy())
            tempseq2=numpy.array([i for i in range(L) if (tempseq1[i]-tempseq[i])==0]) # identify these samples that represent the local maximum data density
            NC=tempseq2.shape[0]
            center=numpy.empty((NC,W))
            sup=numpy.empty((NC,))
            dist2=numpy.zeros((NC,L))
            for ii in range(0,NC):
                 dist2[ii,:]=dist1[tempseq2[ii],:]/numpy.sum(dist1[tempseq2[ii],:])  # decide how to proportionaly assign a data sample to each of these local maxima
                 center[ii,:]=numpy.sum((data0.transpose()*dist2[ii,:]),axis=1)  # get the soft prototypes based on these local maxima as a combination of all existing data samples with different proportions
            sup=numpy.sum(dist2/numpy.sum(dist2,axis=0),axis=1) # get the supports of these soft prototypes
        else: # if the number of data samples received is too small
            NC=L
            center=data0.reshape((L,W))
            sup=numpy.ones((L,))
        return NC,center,sup 
    # the first output: number of the soft prototypes identified from data
    # the second output: soft prototypes
    # the third output: supports of the soft prototypes
    
    def thresholdestimation(self,tempdist): # estimate data-drive treshold from the current input data iteratively
        temp0=numpy.mean(tempdist)
        for ii in range(0,self.granularity):
            temp1=temp0.copy()
            tempdist=tempdist[tempdist<=temp0] # filter out the distances below the average
            temp0=numpy.mean(tempdist) # recalculate the average of the distances between samples
            if temp0==0:
                temp0=temp1.copy()
                break
        return temp0 # the local distance threshold
   
    def prototypeintegration(self,center1,NC1,sup1,kk): # merge the latestly identified soft prototyoes into the existing knowledge base
        seq1=numpy.append(numpy.array(range(0,self.NP[kk],self.chunksize1)),self.NP[kk]) 
        mdist2=numpy.full((NC1,), numpy.inf)
        dist0=numpy.zeros((self.NP[kk],NC1),dtype=float)
        for jj in range(0,seq1.shape[0]-1):
            dist2=self.cdist(self.prototype[kk][range(seq1[jj],seq1[jj+1]),:],center1) # calculate the distances between prototypes in the existing knowledge base and the prototypes learned from the current data chunk
            dist0[seq1[jj]:seq1[jj+1],:]=dist2.copy()
        mdist2=numpy.min(dist0,axis=0)
        seq3=[i for i in range(NC1) if mdist2[i]>self.averdist] # find the new prototypes that are distinctive from existing ones in the knowledge base
        seq4=[i for i in range(NC1) if mdist2[i]<=self.averdist] # find the new prototypes that are not sufficiently distinctive from existing ones in the knowledge base
        self.prototype[kk]=numpy.append(self.prototype[kk],center1[seq3,:],axis=0) # expand the knowledge base with the new distinctive prototypes of the kk th class
        self.support[kk]=numpy.append(self.support[kk],sup1[seq3],axis=0) # add the corresponding supports to the knowledge base
        self.NP[kk]+=sup1[seq3].shape[0] #update the number of prototypes in the knowledge base
        dist3=self.cdist(center1[seq3,:],center1[seq4,:]) # calculate the mutual distances between these distinctive prototypes and less distinctive prototypes 
        center1=center1[seq4,:].copy() 
        sup1=sup1[seq4]
        dist0=numpy.exp(-1*numpy.append(dist0[:,seq4].copy(),dist3,axis=0)/self.averdist) # expand the distance matrix and convert it to the similarity matrix
        dist10=numpy.sum(dist0.copy(),axis=0)
        dist0=dist0.copy()/dist10*sup1
        temps1=numpy.sum(dist0,axis=1)
        self.prototype[kk]=(((self.prototype[kk].transpose()*self.support[kk])+numpy.matmul(dist0,center1).transpose())/(self.support[kk]+temps1)).transpose() # update the latest knowledge base with these less distinctive prototypes using the similarity matrix 
        self.support[kk]+=temps1 # update the supports of the prototypes in the knowledge base
    
    def prototypepruning(self,kk): # fuse the highly similar prototypes together to reduce the size of the knowledge base (reduce redundancy)
        tempthreshold=self.delta*self.averdist # setup the threshold for identifying the pair of prototypes with high spatial similarity
        tempseq=numpy.argsort(self.support[kk]) # sort the prototypes in the knowledge base according to their supports (the pruning process starts with the prototype with the lowest support)
        self.support[kk]=self.support[kk][tempseq,].copy()
        self.prototype[kk]=self.prototype[kk][tempseq,:].copy()
        seq1=numpy.append(numpy.array(range(0,self.NP[kk],self.chunksize1)),self.NP[kk])
        tempNP=numpy.zeros((self.NP[kk],))
        for jj in range(0,seq1.shape[0]-1): # calculate the mutual distances between prototypes and obtain the distance matrix
            dist2=self.pdist(self.prototype[kk][range(seq1[jj],seq1[jj+1]),:])
            dist2=scipy.spatial.distance.squareform(dist2)
            dist3=dist2.copy()
            dist3[dist2<tempthreshold]=1
            dist3[dist2>=tempthreshold]=0
            dist3=numpy.triu(dist3)-numpy.diag(numpy.diag(dist3))
            tempNP[range(seq1[jj],seq1[jj+1])]=numpy.sum(dist3,axis=1)
            for qq in range(jj+1,seq1.shape[0]-1):
                dist2=self.cdist(self.prototype[kk][range(seq1[jj],seq1[jj+1]),:],self.prototype[kk][range(seq1[qq],seq1[qq+1]),:])
                dist3=dist2.copy()
                dist3[dist2<tempthreshold]=1
                dist3[dist2>=tempthreshold]=0   
                tempNP[range(seq1[jj],seq1[jj+1])]=tempNP[range(seq1[jj],seq1[jj+1])]+numpy.sum(dist3,axis=1)
        seq2=[i for i in range(self.NP[kk]) if tempNP[i]>0] # identify these prototypes with lower supports that are spatially similar to prototypes with greater supports
        seq3=[i for i in range(self.NP[kk]) if tempNP[i]==0] # identify these prototypes that are distinctive from others, in particular, prototypes with greater supprots
        if len(seq2)>0: # if there are prototypes needed to be pruned, these prototypes will be removed from the knowledge base and used for updating the parameters of the remaining prototypes in the knowledge base
            center1=self.prototype[kk][seq2,:].copy() # prototypes to be removed
            sup1=self.support[kk][seq2].copy() # the supports of such prototypes
            self.prototype[kk]=self.prototype[kk][seq3,:].copy() # remaining prototypes in the knowledge base
            self.support[kk]=self.support[kk][seq3].copy() # s
            self.NP[kk]=self.prototype[kk].shape[0]
            seq1=numpy.append(numpy.array(range(0,self.NP[kk],self.chunksize1)),self.NP[kk])
            dist0=numpy.zeros((self.NP[kk],center1.shape[0]),dtype=float) 
            for jj in range(0,seq1.shape[0]-1): # calculate the mutual distances between prototypes to be removed and the prototypes remained in the knowledge base
                dist2=self.cdist(self.prototype[kk][range(seq1[jj],seq1[jj+1]),:],center1)
                dist0[seq1[jj]:seq1[jj+1],:]=dist2.copy()
            dist0=numpy.exp(-1*dist0/self.averdist) # convert the distance matrix to a similarity matrix
            dist10=numpy.sum(dist0.copy(),axis=0)
            dist0=dist0.copy()/dist10*sup1
            temps1=numpy.sum(dist0,axis=1)
            self.prototype[kk]=(((self.prototype[kk].transpose()*self.support[kk])+numpy.matmul(dist0,center1).transpose())/(self.support[kk]+temps1)).transpose() # update the knowledge base with these pruned prototypes
            self.support[kk]+=temps1 # update the supports of the remaining prototypes in the knowledge base
    
    def pdist(self,data0): # calculate the distances between data samples within the same set
        temp=scipy.spatial.distance.pdist(data0,'minkowski', p=self.p1)**self.p2/data0.shape[1]
        return temp
    
    def cdist(self,data0,data1): # calculate the pairwise distances between two sets of data samples
        temp=scipy.spatial.distance.cdist(data0,data1,'minkowski', p=self.p1)**self.p2/data0.shape[1]
        return temp    
