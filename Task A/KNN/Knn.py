# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 15:15:38 2018

@author: Aditya Kore
"""
#change later

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
trainY=traindata[0,:]
testX=testdata[1:,:].transpose()
testY=testdata[0,:]



#distance calculations
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

#getting the distances
distance1=[]
distance_dict=[]
classifiermatrix=[]
for i in range(0,len(testX)):
    classifiermatrix.append([])
    
    distance_dict.append({})
    distance1.append([])
    for j in range(0,len(trainX)):
        #key=distance
        #value=label
        distance_dict[i][Euclidean(trainX[j],testX[i])]=trainY[j]#here
        distance1[i].append(Euclidean(trainX[j],testX[i]))
    distance1[i].sort()

#election to select the maximum number of occurence
def election(classifiermatrixarrays):
    d={}
    for elements in classifiermatrixarrays:
        if elements in d:
            d[elements]+=1
        else:
            d[elements]=1
    for k,val in d.items():
        if val == max(d.values()):
            return(int(k))

#taking the k=5

for k in range(len(distance1)):
    #classifiermatrix.append([])
    for l in range(5): #enter the value of k nearest neighbours
        a=distance1[k][l]
        classifiermatrix[k].append(distance_dict[k][a])
finalclassification=[] #final classified elements
for elements in classifiermatrix:
    finalclassification.append(election(elements))
    
#error test and accuracy
error = testY - finalclassification
TestingAccuracy = (1-np.nonzero(error)[0].size/len(error))*100
f=open('Classification output Accuracy','w')
f.write('%.4f Percent'%TestingAccuracy)
f.close()
np.savetxt('Classification output',finalclassification,fmt="%0.0f",delimiter=',')





    

    
    

