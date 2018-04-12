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
    def main(self,a,b):
        s=''
        s=s+str(a)
        s+=':'
        s+=str(b)
        datahandler=dataHandler()
        dataset=datahandler.pickDataClass('ATNTFaceImages400.txt',[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40])
        dataset=dataset.transpose()
        number_per_class=10
        
        testdata,traindata=datahandler.splitData2TestTrain(dataset,number_per_class,s)
        
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
        
        return(prediction,TestingAccuracy)
        
        
        #np.savetxt('Classification output',prediction,fmt="%0.0f")

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
