import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Load in the data and disperse respectively
grad_2030 = pd.read_csv("UCLA_EE_grad_2030.csv",header=None)
grad_2031 = pd.read_csv("UCLA_EE_grad_2031.csv",header=None)

GPA_2030 = grad_2030[0].tolist()
GRE_2030 = grad_2030[1].tolist()
labels_2030 = grad_2030[2].tolist()

GPA_2031 = grad_2031[0].tolist()
GRE_2031 = grad_2031[1].tolist()
labels_2031 = grad_2031[2].tolist()

dataCount = len(labels_2030)

GPA = GPA_2030+GPA_2031
GRE = GRE_2030+GRE_2031
labels = labels_2030+labels_2031

#Create our Testing and Training Data
testingCount = 40
testingGPA = np.zeros(testingCount)
testingGRE = np.zeros(testingCount)
testingLabels = np.zeros(testingCount)
for i in range (testingCount):
    testingGPA[i] = GPA[i]
    testingGRE[i] = GRE[i]
    testingLabels[i] = labels[i]

trainingCount = 160
trainingGPA = np.zeros(trainingCount)
trainingGRE = np.zeros(trainingCount)
trainingLabels = np.zeros(trainingCount)
for i in range (trainingCount):
    index = testingCount + i
    trainingGPA[i] = GPA[index]
    trainingGRE[i] = GRE[index]
    trainingLabels[i] = labels[index]

#Iterate through all the k's
kList = [1,2,3,4,5,6,7,8,9,10,11,12]
numK = len(kList)

totalDistTrain = np.zeros((trainingCount,trainingCount))
totalDistTest = np.zeros((testingCount,trainingCount))

accuracyTrain = np.zeros((numK,trainingCount))
accuracyTest = np.zeros((numK,testingCount))

#Set our ytie value
ytie = 0

#Start with the training accuracy
for test in range (trainingCount):
    xTestGPA = trainingGPA[test]
    xTestGRE = trainingGRE[test]
    
    #Determine distance of each training point
    for train in range (trainingCount):
        if(test == train):
            continue
        xTrainGPA = trainingGPA[train]
        xTrainGRE = trainingGRE[train]
        totalDistTrain[test,train] = np.sqrt((xTestGPA-xTrainGPA)**2+(xTestGRE-xTrainGRE)**2)
    
    #Use that distance matrix with each of the k values to produce a correct/incorrect response via k
    for kTemp in range (numK):
        k = kList[kTemp]
        tempDist = totalDistTrain[test].tolist()

        indexList = []

        for neighbor in range (k):
            index = np.argmin(tempDist)
            tempDist.pop(index)
            indexList.append(index)

        #Easiest way is to find the average of their labels and use 0.5 as the threshold
        labelList = []
        for i in range (k):
            labelList.append(trainingLabels[indexList[i]])

        avg = sum(labelList)/len(labelList)

        if(avg > 0.5):
            result = 1
        elif(avg < 0.5):
            result = 0
        else:
            if(ytie == 0):
                result = 0
            else:
                result = 1

        if(result == trainingLabels[test]):
            acc = 1
        else:
            acc = 0

        accuracyTrain[kTemp,test] = acc

#Calculate our training accuracy
trainingAccuracy = accuracyTrain.sum(axis=1)/trainingCount*100
print(trainingAccuracy)

#Now for testing accuracy
#Iterate among each test point
for test in range (testingCount):
    xTestGPA = testingGPA[test]
    xTestGRE = testingGRE[test]
    
    #Determine distance of each training point
    for train in range (trainingCount):
        xTrainGPA = trainingGPA[train]
        xTrainGRE = trainingGRE[train]
        totalDistTest[test,train] = np.sqrt((xTestGPA-xTrainGPA)**2+(xTestGRE-xTrainGRE)**2)
    
    #Use that distance matrix with each of the k values to produce a correct/incorrect response via k
    for kTemp in range (numK):
        k = kList[kTemp]
        tempDist = totalDistTest[test].tolist()

        indexList = []

        for neighbor in range (k):
            index = np.argmin(tempDist)
            tempDist.pop(index)
            indexList.append(index)

        #Easiest way is to find the average of their labels and use 0.5 as the threshold
        labelList = []
        for i in range (k):
            labelList.append(trainingLabels[indexList[i]])

        avg = sum(labelList)/len(labelList)

        if(avg > 0.5):
            result = 1
        elif(avg < 0.5):
            result = 0
        else:
            if(ytie == 0):
                result = 0
            else:
                result = 1

        if(result == testingLabels[test]):
            acc = 1
        else:
            acc = 0

        accuracyTest[kTemp,test] = acc

#Calculate our testing accuracy
testingAccuracy = accuracyTest.sum(axis=1)/testingCount*100
print(testingAccuracy)

#Time to plot the accuracies
plt.plot(kList,trainingAccuracy,label="Training Accuracy")
plt.plot(kList,testingAccuracy,label="Testing Accuracy")
plt.xlabel("k Value")
plt.ylabel("Accuracy (%)")
plt.title("Training and Testing Accuracy vs. k Value with y_tie = 0")
plt.legend()
plt.show()
