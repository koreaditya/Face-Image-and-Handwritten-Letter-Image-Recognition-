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
        #here q is the dataset after selecting classes from the dataset
        
    def splitData2TestTrain(self,filename, number_per_class, test_instance):
        #the arguments given in this function are for the testing data. training data is calculated by removing testing data from the dataset
        test_instance=test_instance.split(':')
        self.lead=int(test_instance[0])
        self.trail=int(test_instance[1])
        testdata=[]
        traindata=[]
        #returns test and train data
        
        
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
    #use only if there are letters    
    def letter_2_digit_convert(self,string):
        self.array = []
        for x in string:
            self.array.append(ord(x)-64)
        self.array=np.stack(self.array)
        return(self.array)
        

    