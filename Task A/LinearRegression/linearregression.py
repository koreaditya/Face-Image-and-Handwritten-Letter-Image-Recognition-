# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 17:32:13 2018

@author: Aditya Kore

"""

import numpy as np
class dataHandler(object):
    def pickDataClass(self,filename,class_ids):
        filename1=np.loadtxt(filename,delimiter=',')
        finaldata=[]
        self.lenclassids=len(class_ids)
        for i in range (len(filename1[0])):
            if filename1[0][i] in class_ids:
                finaldata.append(filename1[:,i])
            else:
                continue
            q=np.stack(finaldata)        
        return q
    
    def splitData2TestTrain(self,filename, number_per_class, test_instance):
        test_instance=test_instance.split(':')
        self.lead=int(test_instance[0])
        self.trail=int(test_instance[1])
        self.lead1=int(test_instance[0])
        self.trail1=int(test_instance[1])
        testdata=[]
        traindata=[]
        
        max=number_per_class
        min=0
        while max<=len(filename[0]):
            testdata.append(filename[:,self.lead:self.trail])
            traindata.append(filename[:,min:self.lead])
            traindata.append(filename[:,self.trail:max])
            self.lead+=number_per_class
            self.trail+=number_per_class
            max+=number_per_class
            min+=number_per_class
            
            
        testdata=np.hstack(testdata)
        traindata=np.hstack(traindata)
        return(testdata,traindata)  
    def letter_2_digit_convert(self,string):
        self.array = []
        for x in string:
            self.array.append(ord(x)-64)
        self.array=np.stack(self.array)
        return(self.array)
datahandler=dataHandler()
dataset=datahandler.pickDataClass('HandWrittenLetters.txt',datahandler.letter_2_digit_convert('ABCDE'))
dataset=dataset.transpose()
number_per_class=39
testdata,traindata=datahandler.splitData2TestTrain(dataset,number_per_class,'30:39')#first 30 training last 9 testing


trainX=traindata[1:,:]
trainY=traindata[0,:,None]
testX=testdata[1:,:]
testY=testdata[0,:]


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_Y = LabelEncoder()
trainY[:,0]=labelencoder_Y.fit_transform(trainY[:,0])
onehotencoder=OneHotEncoder(categorical_features=[0])
trainY=onehotencoder.fit_transform(trainY).toarray()
trainY=trainY.transpose()
#Xtest=testdata[:,:]
#Ytest=testdataY[0,:]


A_train=np.ones((1,len(trainX[0])))
A_test=np.ones((1,len(testY)))


Xtrain_padding = np.row_stack((trainX,A_train))
Xtest_padding = np.row_stack((testX,A_test))

B_padding = np.dot(np.linalg.pinv(Xtrain_padding.T), trainY.T)
Ytest_padding = np.dot(B_padding.T,Xtest_padding)
Ytest_padding_argmax = np.argmax(Ytest_padding,axis=0)+1
err_test_padding = testY - Ytest_padding_argmax
TestingAccuracy_padding = (1-np.nonzero(err_test_padding)[0].size/len(err_test_padding))*100
f=open('Classification output Accuracy','w')
f.write('%.4f Percent'%TestingAccuracy_padding)
f.close()
np.savetxt('Classification output',Ytest_padding_argmax,fmt="%0.0f",delimiter=',')