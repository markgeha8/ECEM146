import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Load in the data and disperse respectively
regTrain = pd.read_csv("regression_train.csv",header=None)
regTest = pd.read_csv("regression_test.csv",header=None)

inTrain = regTrain[0]
outTrain = regTest[1]
inTest = regTest[0]
outTest = regTest[1]

dataPoints = len(inTrain)

#Plot the training data
plt.scatter(inTrain,outTrain,marker='x',color='k')
plt.xlabel("Input Value")
plt.ylabel("Output Value")
plt.title("Input vs. Output Raw Scatter Plot")
plt.show()

#Calculate linear regression points
y = outTrain
X = np.zeros((dataPoints,2)) #X is 20x2
for i in range (dataPoints):
    X[i][0] = 1
    X[i][1] = inTrain[i]

#Calculate the weight vector
XTX = X.T.dot(X)
inverse = np.linalg.inv(XTX)
XTy = X.T.dot(y)
wStar = inverse.dot(XTy)

#Calculate our loss function
Jw = 0
for i in range (dataPoints):
    Jw += (X[i].dot(wStar)-y[i])**2

print("w* =", wStar)
print("J(w) =", Jw)

#Plot assumed model
model = X.dot(wStar)
plt.scatter(inTrain,outTrain,marker='x',color='k')
plt.plot(inTrain,model,color='r')
plt.xlabel("Input Value")
plt.ylabel("Output Value")
plt.title("Input vs. Output with Linear Regression")
plt.show()

#Gradient Descent
maxIter = 10000
thresh = 0.0001

LR = [0.05, 0.001, 0.0001, 0.00001]

iterTotal = []
JwTotal = []

for lr in (LR):
    J_t = 100000
    J_tm1 = 0
    iter = 0
    
    #Initialize with 0s for linear regression (2x1)
    wGD = np.zeros((2,1))

    print("LR being tested is",lr)

    while(iter < maxIter and (abs(J_t - J_tm1) > thresh)):
        #Used to determine difference
        J_tm1 = J_t

        #Run algorithm with updated wGD values
        diff = [0,0]
        for i in range (dataPoints):
            diff[0] += (X[i].dot(wGD)-y[i])*X[i][0]
            diff[1] += (X[i].dot(wGD)-y[i])*X[i][1]

        wGD[0] -= lr*diff[0]
        wGD[1] -= lr*diff[1]

        #Update J(w) for testing
        J_t = 0
        for i in range (dataPoints):
            J_t += (X[i].dot(wGD)-y[i])**2
        
        iter += 1
    
    iterTotal.append(iter)
    JwTotal.append(J_t[0])
    print("Iterations Required:",iter)
    print("J(w) Accumulated:",J_t)
    print()

print("Learning Rates Tested:",LR)
print("Total Iterations per LR:",iterTotal)
print("J(w) per LR:",JwTotal)

print()
print("_______________________________________________________________________")
print()

#Visual Gradient Descent
lr = 0.05
maxIter = 40
J_t = 100000
J_tm1 = 0
thresh = 0.0001

iter = 0

wGD = np.zeros((2,1))

w0Final = []
w1Final = []
JwFinal = []

while(iter <= maxIter and (abs(J_t - J_tm1) > thresh)):
        #Used to determine difference
        J_tm1 = J_t
        #Run algorithm with updated wGD values
        diff = [0,0]
        for i in range (dataPoints):
            diff[0] += (X[i].dot(wGD)-y[i])*X[i][0]
            diff[1] += (X[i].dot(wGD)-y[i])*X[i][1]

        wGD[0] -= lr*diff[0]
        wGD[1] -= lr*diff[1]

        #Update J(w) for testing
        J_t = 0
        for i in range (dataPoints):
            J_t += (X[i].dot(wGD)-y[i])**2
        
        if((iter%10) == 0):
            w0Final.append(wGD[0][0])
            w1Final.append(wGD[1][0])
            JwFinal.append(J_t[0])

        iter += 1

print("Incremented Weights[0] =",w0Final)
print("Incremented Weights[1] =",w1Final)
print("Incremented Loss =",JwFinal)

#Produce the plots
wTemp = np.zeros((2,1))

wTemp[0] = w0Final[0]
wTemp[1] = w1Final[0]
model_0 = X.dot(wTemp)

wTemp[0] = w0Final[1]
wTemp[1] = w1Final[1]
model_10 = X.dot(wTemp)

wTemp[0] = w0Final[2]
wTemp[1] = w1Final[2]
model_20 = X.dot(wTemp)

wTemp[0] = w0Final[3]
wTemp[1] = w1Final[3]
model_30 = X.dot(wTemp)

wTemp[0] = w0Final[4]
wTemp[1] = w1Final[4]
model_40 = X.dot(wTemp)

#Overlay all the plots
plt.scatter(inTrain,outTrain,marker='x',color='k')
plt.plot(inTrain,model_0,color='r')
plt.plot(inTrain,model_10,color='b')
plt.plot(inTrain,model_20,color='g')
plt.plot(inTrain,model_30,color='y')
plt.plot(inTrain,model_40,color='c')
plt.legend(["0 Iter","10 Iter","20 Iter","30 Iter","40 Iter","Raw Data"],loc ="lower right")
plt.xlabel("Input Value")
plt.ylabel("Output Value")
plt.title("Input vs. Output with Linear Regression")
plt.show()

#Polynomial Regression
#Create our power vectors and calculate desired phi vectors
power_0 = np.zeros((dataPoints))
power_1 = power_0 + np.ones_like(power_0)
power_2 = power_1 + np.ones_like(power_0)
power_3 = power_2 + np.ones_like(power_0)
power_4 = power_3 + np.ones_like(power_0)
power_5 = power_4 + np.ones_like(power_0)
power_6 = power_5 + np.ones_like(power_0)
power_7 = power_6 + np.ones_like(power_0)
power_8 = power_7 + np.ones_like(power_0)
power_9 = power_8 + np.ones_like(power_0)
power_10 = power_9 + np.ones_like(power_0)

col0 = np.power(inTrain,power_0)
col1 = np.power(inTrain,power_1)
col2 = np.power(inTrain,power_2)
col3 = np.power(inTrain,power_3)
col4 = np.power(inTrain,power_4)
col5 = np.power(inTrain,power_5)
col6 = np.power(inTrain,power_6)
col7 = np.power(inTrain,power_7)
col8 = np.power(inTrain,power_8)
col9 = np.power(inTrain,power_9)
col10 = np.power(inTrain,power_10)

phiTrain0 = np.c_[col0]
phiTrain1 = np.c_[col0,col1]
phiTrain2 = np.c_[col0,col1,col2]
phiTrain3 = np.c_[col0,col1,col2,col3]
phiTrain4 = np.c_[col0,col1,col2,col3,col4]
phiTrain5 = np.c_[col0,col1,col2,col3,col4,col5]
phiTrain6 = np.c_[col0,col1,col2,col3,col4,col5,col6]
phiTrain7 = np.c_[col0,col1,col2,col3,col4,col5,col6,col7]
phiTrain8 = np.c_[col0,col1,col2,col3,col4,col5,col6,col7,col8]
phiTrain9 = np.c_[col0,col1,col2,col3,col4,col5,col6,col7,col8,col9]
phiTrain10 = np.c_[col0,col1,col2,col3,col4,col5,col6,col7,col8,col9,col10]

col0 = np.power(inTest,power_0)
col1 = np.power(inTest,power_1)
col2 = np.power(inTest,power_2)
col3 = np.power(inTest,power_3)
col4 = np.power(inTest,power_4)
col5 = np.power(inTest,power_5)
col6 = np.power(inTest,power_6)
col7 = np.power(inTest,power_7)
col8 = np.power(inTest,power_8)
col9 = np.power(inTest,power_9)
col10 = np.power(inTest,power_10)

phiTest0 = np.c_[col0]
phiTest1 = np.c_[col0,col1]
phiTest2 = np.c_[col0,col1,col2]
phiTest3 = np.c_[col0,col1,col2,col3]
phiTest4 = np.c_[col0,col1,col2,col3,col4]
phiTest5 = np.c_[col0,col1,col2,col3,col4,col5]
phiTest6 = np.c_[col0,col1,col2,col3,col4,col5,col6]
phiTest7 = np.c_[col0,col1,col2,col3,col4,col5,col6,col7]
phiTest8 = np.c_[col0,col1,col2,col3,col4,col5,col6,col7,col8]
phiTest9 = np.c_[col0,col1,col2,col3,col4,col5,col6,col7,col8,col9]
phiTest10 = np.c_[col0,col1,col2,col3,col4,col5,col6,col7,col8,col9,col10]

#Start running algorithms to determine best algorithm
phiTrain = [phiTrain0,phiTrain1,phiTrain2,phiTrain3,phiTrain4,phiTrain5,phiTrain6,phiTrain7,phiTrain8,phiTrain9,phiTrain10]
phiTest = [phiTest0,phiTest1,phiTest2,phiTest3,phiTest4,phiTest5,phiTest6,phiTest7,phiTest8,phiTest9,phiTest10]
yTrain = outTrain
yTest = outTest
JTrain = []
JTest = []
wTrain = []

print("Starting Training Data")
for phi in (phiTrain):
    #Calculate the weight vector
    PTP = phi.T.dot(phi)
    inverse = np.linalg.inv(PTP)
    PTy = phi.T.dot(yTrain)
    wStar = inverse.dot(PTy)

    #Calculate our loss function
    Jw = 0
    for i in range (dataPoints):
        Jw += (phi[i].dot(wStar)-yTrain[i])**2

    wTrain.append(wStar)
    JTrain.append(Jw)

print()

trainMinJ = min(JTrain)
trainInd = JTrain.index(trainMinJ)
trainRMSE = np.sqrt(trainMinJ/dataPoints)
print("Best Training Model is when m =", trainInd)

#Using the 10th model's wopt
wopt = wTrain[trainInd]

print()
print("Starting Testing Data")
j = 0

for phi in (phiTest):
    #Utilize the same weight vector as before
    #PTP = phi.T.dot(phi)
    #inverse = np.linalg.inv(PTP)
    #PTy = phi.T.dot(yTest)
    #wStar = inverse.dot(PTy)

    wStar = wTrain[j]

    #Calculate our loss function
    Jw = 0
    for i in range (dataPoints):
        Jw += (phi[i].dot(wStar)-yTest[i])**2

    JTest.append(Jw)
    
    j += 1
print()

trainMinJ = min(JTrain)
trainInd = JTrain.index(trainMinJ)
trainRMSE = np.sqrt(trainMinJ/dataPoints)
print("Best Training Model is when m =", trainInd)
print("The Minimum Value is",trainMinJ)
print("Training RMSE =",trainRMSE)

print()

testInd = trainInd
testMinJ = JTest[int(testInd)]
testRMSE = np.sqrt(testMinJ/dataPoints)
print("Best Testing Model is when m =", testInd)
print("The Value is",testMinJ)
print("Testing RMSE =",testRMSE)

m = [0,1,2,3,4,5,6,7,8,9,10]
trainRMSETotal = []
testRMSETotal = []

#Calculate RMSE of every J(w) point
for i in range (len(m)):
    trainRMSETotal.append(np.sqrt(JTrain[i]/dataPoints))
    testRMSETotal.append(np.sqrt(JTest[i]/dataPoints))

plt.plot(m,trainRMSETotal,color='r')
plt.plot(m,testRMSETotal,color='b')
plt.legend(["Training","Testing"],loc ="upper left")
plt.xlabel("Model Complexity, m")
plt.ylabel("RMSE")
plt.title("Model Complexity vs. RMSE")
plt.show()