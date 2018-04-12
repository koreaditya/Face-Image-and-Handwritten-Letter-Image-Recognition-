# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 16:12:25 2018

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
dataset=datahandler.pickDataClass('HandWrittenLetters.txt',datahandler.letter_2_digit_convert('ABCDE'))#
dataset=dataset.transpose()
number_per_class=39
testdata,traindata=datahandler.splitData2TestTrain(dataset,number_per_class,'30:39')#first 30 training last 9 testing
                                                    
trainX=traindata[1:,:].transpose()
trainY=traindata[0,:]
testX=testdata[1:,:].transpose()
testY=testdata[0,:]

from sklearn.svm import SVC
classifier=SVC(kernel='linear')
classifier.fit(trainX,trainY)

prediction=classifier.predict(testX)

error = testY - prediction
TestingAccuracy = (1-np.nonzero(error)[0].size/len(error))*100 
f=open('Classification output Accuracy','w')
f.write('%.4f Percent'%TestingAccuracy)
f.close()
np.savetxt('Classification output',prediction,fmt="%0.0f",delimiter=',')