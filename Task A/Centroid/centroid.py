# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 14:17:52 2018

@author: Aditya Kore
"""
"""numpy used only to load data and write data to the file"""
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
    #use only in there are letters    
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
                                                    

trainX=traindata[1:,:].transpose()
trainY=traindata[0,:,None]
testX=testdata[1:,:].transpose()
testY=testdata[0,:]


def Euclidean(A_train,B_test):
    train_i=0
    test_i=0
    distance=0
    while(train_i<len(A_train) and test_i<len(B_test)):
        distance=distance+(((A_train[train_i]-B_test[test_i]))**2)
        train_i+=1
        test_i+=1
    distance=distance**0.5    
    distance=('%.2f' % distance)
    return float(distance)

#Centroid
centroidList={}
def Centroid(trainXdata,trainYdata):
    
    centroidListArray=[]
    for o in range (0,len(trainXdata[0])):
        sum=0
        for p in range(len(trainXdata)):
            sum+=(trainXdata[p][o])
        centroidListArray.append(float('%.2f'%(sum/len(trainXdata))))
    a=trainYdata[0]
    centroidList[a]=centroidListArray
    
#change indexes



#indices changes according to the training sets
ele=0
count=0
#divide by number of labels

for cen in range(datahandler.lenclassids-1):
    while trainY[ele]==trainY[ele+1]:
        ele+=1
    Centroid(trainX[count:ele+1,0:],trainY[ele]) #number each label
    ele+=1
    count=ele
Centroid(trainX[ele:,0:],trainY[-1])


cdistances=[]#distances of each testing instances from the centroids
for k in range(len(testX)):
    cdistances.append({})
    for elements in centroidList:
        cdistances[k][elements]=Euclidean(centroidList[elements],testX[k])

finalclassifier=[]
for elements in cdistances:
    leastdistance=min(elements.values())
    for key,val in elements.items():
        if val==leastdistance:
            finalclassifier.append(key)
            break

#error test and accuracy
error = testY - finalclassifier
TestingAccuracy = (1-np.nonzero(error)[0].size/len(error))*100   
f=open('Classification output Accuracy','w')
f.write('%.4f Percent'%TestingAccuracy)
f.close()
np.savetxt('Classification output',finalclassifier,fmt="%0.0f",delimiter=',')

       
       
       
       
       
