import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

#Load in the data and disperse respectively
MNIST5 = pd.read_csv("MNIST5.csv",header=None)

#Load data into easily accessible array (image as first index, pixel as second index)
pixels = 784
dataCount = 400
dim = 28

data = np.zeros((dataCount,pixels))

for i in range (dataCount):
    for j in range (pixels):
        data[i][j] = MNIST5[j][i]

#Our X is based on each row of data vector, so mean should be of dimension pixels
dataMean = np.mean(data,axis=0)
dataAdjust = data-dataMean

#Calculate covariance matrix given shifted data
S = dataAdjust.T.dot(dataAdjust)/dataCount

#Our eigenvalues are in ascending order right now, so we need to read from back to front for descending
minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 255))
eVals, eVecs = np.linalg.eigh(S)

maxValue = 100
xAxis = np.linspace(1,100,num=100)
eVals100 = []
eVecs100 = []

for i in range(maxValue):
    eVals100.append(eVals[pixels-1-i])
    eVecs100.append(eVecs[pixels-1-i])

plt.scatter(xAxis,eVals100)
plt.xlabel("Eigenvalue Number")
plt.ylabel("Eigenvalue")
plt.title("100 Largest Eigenvalues")
plt.show()

#Visualization of Vectors
visCount = 4

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, visCount)
fig.suptitle('First Four Eigenvectors')

for i in range(visCount):
    eVecVisTemp = eVecs100[i].reshape(dim,dim)

    #Scale from 0 to 255
    eVecVis = minmax_scale.fit_transform(eVecVisTemp)

    if(i == 0):
        ax1.imshow(eVecVis)
        ax1.set_title("1st")
    elif(i == 1):
        ax2.imshow(eVecVis)
        ax2.set_title("2nd")
    elif(i == 2):
        ax3.imshow(eVecVis)
        ax3.set_title("3rd")
    else:
        ax4.imshow(eVecVis)
        ax4.set_title("4th")

plt.show()

#Plot when M = 0 as a ground truth
plt.imshow(dataMean.reshape(28,28))
plt.title("M = 0")
plt.show()

#To compress the top image, let us utilize the dataset
compCount = 5
firstIm = data[0]

M = [1,10,50,250]

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, compCount)
fig.suptitle('Resulting Images from Differing M Values')

#Cycle through our M's and utilize the equation that we were provided
for m in (M):
    print("For M =",m)
    x_pred = dataMean

    for i in range(m):
        ui = eVecs[pixels-1-i]
        projectionIm = firstIm.T.dot(ui)
        projectionMean = dataMean.dot(ui)

        x_pred += (projectionIm - projectionMean)*ui
    
    x_pred28 = x_pred.reshape(28,28)
    print(x_pred28)

    if(m == 1):
        ax2.imshow(x_pred28)
        ax2.set_title("M = 1")
    elif(m == 10):
        ax3.imshow(x_pred28)
        ax3.set_title("M = 10")
    elif(m == 50):
        ax4.imshow(x_pred28)
        ax4.set_title("M = 50")
    else:
        ax5.imshow(x_pred28)
        ax5.set_title("M = 250")

ax1.imshow(firstIm.reshape(28,28))
ax1.set_title("Original")
plt.show()