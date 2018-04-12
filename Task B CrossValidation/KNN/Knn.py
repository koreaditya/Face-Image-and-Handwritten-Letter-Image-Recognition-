# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 15:15:38 2018

@author: Aditya Kore
"""

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
        
    def main(self,a,b):
        s=''
        s=s+str(a)
        s+=':'
        s+=str(b)
        datahandler=dataHandler()
        dataset=datahandler.pickDataClass('ATNTFaceImages400.txt',[1,2,3,4,5])
        dataset=dataset.transpose()
        number_per_class=10
        testdata,traindata=datahandler.splitData2TestTrain(dataset,number_per_class,s)#first 30 training last 9 testing
        
        
        
        trainX=traindata[1:,:].transpose()
        trainY=traindata[0,:]
        testX=testdata[1:,:].transpose()
        testY=testdata[0,:]



        #distance calculations
        
        
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
        
        
        #taking the k=3
        
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
        return(finalclassification,TestingAccuracy)
a=dataHandler()
classification1,accuracy1=a.main(0,2)
np.savetxt('Classification output 1',classification1,fmt="%0.0f")
f1=open('Classification output Accuracy 1','w')
f1.write('%f'%accuracy1)
f1.close()
classification2,accuracy2=a.main(2,4)
np.savetxt('Classification output 2',classification2,fmt="%0.0f")
f2=open('Classification output Accuracy 2','w')
f2.write('%f'%accuracy2)
f2.close()
classification3,accuracy3=a.main(4,6)
np.savetxt('Classification output 3',classification3,fmt="%0.0f")
f3=open('Classification output Accuracy 3','w')
f3.write('%f'%accuracy3)
f3.close()
classification4,accuracy4=a.main(6,8)
np.savetxt('Classification output 4',classification4,fmt="%0.0f")
f4=open('Classification output Accuracy 4','w')
f4.write('%f'%accuracy4)
f4.close()
classification5,accuracy5=a.main(8,10)
np.savetxt('Classification output 5',classification5,fmt="%0.0f")
f5=open('Classification output Accuracy 5','w')
f5.write('%f'%accuracy5)
f5.close()
AverageAccuracy=(accuracy1+accuracy2+accuracy3+accuracy4+accuracy5)/5
f6=open('AverageAccuracy','w')
f6.write('%f'%AverageAccuracy)
f6.close()



    

    
    

