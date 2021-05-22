import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import scipy.stats as stats

#Load and print photo
image = cv2.imread('UCLA_Bruin.jpg')
plt.imshow(image)
plt.show()

#Number of clusters
def kmAlg(image,KMax):
    means = np.zeros((KMax,3))
    horiz = 400
    vert = 300

    print("Calculating Initialization Means")

    #Determining initial means
    #Start with upper left pixel as a mean of first cluster
    means[0] = image[0][0]

    for k in range (KMax):
        minCoor = [0,0]
        tempDists = np.zeros(k+1)
        minDist = np.zeros(k+1)

        if(not(k == 0)):
            for i in range (vert):
                for j in range (horiz):
                    for tempK in range (k+1):
                        dist = (image[i][j] - means[tempK]).T.dot((image[i][j] - means[tempK]))
                        tempDists[tempK] = dist

                    if(tempDists.min() > minDist.min()):
                        minCoor = [i,j]
                        for tempK in range (k+1):
                            minDist[tempK] = tempDists[tempK]

            means[k] = image[minCoor[0],minCoor[1]]

    print("Means:")
    print(means)

    #Commence K-Means Algorithm
    print("_______________________________")
    print("Commencing Algorithm")
    print()

    maxIter = 10
    J = np.zeros(maxIter)
    cluster = np.zeros((vert,horiz))

    for iter in range (maxIter):
        #Assign cluster labels to each point
        for i in range (vert):
            for j in range (horiz):
                dists = np.zeros(KMax)

                for k in range (KMax):
                        dist = (image[i][j] - means[k]).T.dot((image[i][j] - means[k]))
                        dists[k] = dist
                
                clust = np.argmin(dists)
                cluster[i][j] = clust

        #Update means
        for k in range (KMax):
            newMean = np.zeros(3)
            totalClust = 0
            
            for i in range (vert):
                for j in range (horiz):
                    if(cluster[i][j] == k):
                        newMean += image[i][j]
                        totalClust += 1

            newMean /= totalClust
            means[k] = newMean
        
        print("Iteration:",iter+1)
        print("New Means:")
        print(means)

        #Calculate J
        jTemp = 0

        for k in range(KMax):
            for i in range (vert):
                for j in range (horiz):
                    #Essentially only adding those in the cluster is our rnk = 1
                    if(cluster[i][j] == k):
                        jTemp += (image[i][j]-means[k]).T.dot(image[i][j]-means[k])

        J[iter] = jTemp
        print("Objective Function, J =",jTemp)
        print()

    iterList = np.arange(1,11)
    plt.plot(iterList,J)
    plt.xlabel("Iterations")
    plt.ylabel("Objective Function, J")
    plt.title("Objective Function, J vs. Iterations")
    plt.show()

    tempImage = np.zeros_like(image)
    #Compress the image
    for i in range (vert):
        for j in range (horiz):
            clusterCount = int(cluster[i][j])
            tempImage[i][j] = means[clusterCount]

    plt.imshow(tempImage)
    plt.show()


#Run Algorithm with Different K Values
print("K = 4")
print("_____________________________")
kmAlg(image,4)
print("K = 8")
print("_____________________________")
kmAlg(image,8)
print("K = 16")
print("_____________________________")
kmAlg(image,16)