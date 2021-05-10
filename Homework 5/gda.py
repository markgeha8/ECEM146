import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

#Load in the data and disperse respectively
grad_2030 = pd.read_csv("UCLA_EE_grad_2030_0.csv",header=None)

GPA_2030 = grad_2030[0].tolist()
GRE_2030 = grad_2030[1].tolist()
labels_2030 = grad_2030[2].tolist()

dataCount = len(labels_2030)

GPA_2030_y = []
GPA_2030_n = []
GRE_2030_y = []
GRE_2030_n = []
yesCount = 0
noCount = 0

for i in range (dataCount):
    if(labels_2030[i] == 0):
        GPA_2030_n.append(GPA_2030[i])
        GRE_2030_n.append(GRE_2030[i])
        noCount += 1
    else:
        GPA_2030_y.append(GPA_2030[i])
        GRE_2030_y.append(GRE_2030[i])
        yesCount += 1

#Calculate parameters
#Priors based on Bernoulli
py_0 = noCount/dataCount

#Mean is simply mean of each group
u0_gpa = np.sum(GPA_2030_n)/noCount
u0_gre = np.sum(GRE_2030_n)/noCount
u1_gpa = np.sum(GPA_2030_y)/yesCount
u1_gre = np.sum(GRE_2030_y)/yesCount

#Variance when they're equal is slightly more complicated
varYesGPA = 0
varYesGRE = 0
varNoGPA = 0
varNoGRE = 0

for i in range (yesCount):
    varYesGPA += (GPA_2030_y[i]-u1_gpa)**2
    varYesGRE += (GRE_2030_y[i]-u1_gre)**2

for i in range (noCount):
    varNoGPA += (GPA_2030_n[i]-u0_gpa)**2
    varNoGRE += (GRE_2030_n[i]-u0_gre)**2

#Equal variances
sigEqualGPA = (noCount/dataCount)*(varNoGPA/noCount)+(yesCount/dataCount)*(varYesGPA/yesCount)
sigEqualGRE = (noCount/dataCount)*(varNoGRE/noCount)+(yesCount/dataCount)*(varYesGRE/yesCount)

#Unequal variances
sigGPAy = varYesGPA/yesCount
sigGREy = varYesGRE/yesCount
sigGPAn = varNoGPA/noCount
sigGREn = varNoGRE/noCount

#Decision Boundaries
#Equal variances
bGPAequal = -(sigEqualGPA/(u1_gpa-u0_gpa))*(np.log((1-py_0)/py_0)+(u0_gpa**2-u1_gpa**2)/(2*sigEqualGPA))
bGREequal = -(sigEqualGRE/(u1_gre-u0_gre))*(np.log((1-py_0)/py_0)+(u0_gre**2-u1_gre**2)/(2*sigEqualGRE))

print("GPA2030 Statistics")
print("__________________")
print("Prior Probability of Non-Attending =",py_0)
print("u_o =",u0_gpa)
print("u_1 =",u1_gpa)
print("Equal Variance =",sigEqualGPA)
print("Variance of Admitted =",sigGPAy)
print("Variance of Non-Admitted =",sigGPAn)
print("Equal Decision Boundary, b =",bGPAequal)
print()
print("GRE2030 Statistics")
print("__________________")
print("Prior Probability of Non-Attending =",py_0)
print("u_o =",u0_gre)
print("u_1 =",u1_gre)
print("Equal Variance =",sigEqualGRE)
print("Variance of Admitted =",sigGREy)
print("Variance of Non-Admitted =",sigGREn)
print("Equal Decision Boundary, b =",bGREequal)

print()
print("______________________________________________")
print("Testing Boundary Conditions")

#If our x > b, then it's in class 1 (yes). If our x < b, then it's in class 0 (no)
accuracyGPAEqual = 0
accuracyGREEqual = 0
for i in range (yesCount):
    if(GPA_2030_y[i] >= bGPAequal):
        accuracyGPAEqual += 1

    if(GRE_2030_y[i] >= bGREequal):
        accuracyGREEqual += 1

for i in range (noCount):
    if(GPA_2030_n[i] < bGPAequal):
        accuracyGPAEqual += 1
    
    if(GRE_2030_n[i] < bGREequal):
        accuracyGREEqual += 1

accuracyGPAEqual *= 100/dataCount
accuracyGREEqual *= 100/dataCount

print("Equal Variance GPA Accuracy =",accuracyGPAEqual)
print("Equal Variance GRE Accuracy =",accuracyGREEqual)

a = [1/sigGPAy-1/sigGPAn,1/sigGREy-1/sigGREn]
b = [-2*(u1_gpa/sigGPAy-u0_gpa/sigGPAn),-2*(u1_gre/sigGREy-u0_gre/sigGREn)]
c = [(u1_gpa**2/sigGPAy-u0_gpa**2/sigGPAn)+np.log((1-py_0)*np.sqrt(sigGPAn/sigGPAy)/(py_0)),(u1_gre**2/sigGREy-u0_gre**2/sigGREn)+np.log((1-py_0)*np.sqrt(sigGREn/sigGREy)/(py_0))]

root1 = np.zeros(2)
root2 = np.zeros(2)

for i in range(2):
    #Calculate our roots
    d = (b[i]**2) - (4*a[i]*c[i])

    # find two solutions
    root1[i] = (-b[i]-np.sqrt(d))/(2*a[i])
    root2[i] = (-b[i]+np.sqrt(d))/(2*a[i])

print("The roots for our GPA are:",root1[0],"and",root2[0])
print("The roots for our GRE are:",root1[1],"and",root2[1])

#To test the accuracies of these
#If our x > b, then it's in class 1 (yes). If our x < b, then it's in class 0 (no)
accuracyGPAUnequal = 0
accuracyGREUnequal = 0
for i in range (yesCount):
    if((GPA_2030_y[i] >= root1[0] and GPA_2030_y[i] < root2[0]) or (GPA_2030_y[i] <= root1[0] and GPA_2030_y[i] > root2[0])):
        accuracyGPAUnequal += 1

    if((GRE_2030_y[i] >= root1[1] and GRE_2030_y[i] < root2[1]) or (GRE_2030_y[i] <= root1[1] and GRE_2030_y[i] > root2[1])):
        accuracyGREUnequal += 1

for i in range (noCount):
    if(not((GPA_2030_n[i] >= root1[0] and GPA_2030_n[i] < root2[0]) or (GPA_2030_n[i] <= root1[0] and GPA_2030_n[i] > root2[0]))):
        accuracyGPAUnequal += 1
    
    if(not((GRE_2030_n[i] >= root1[1] and GRE_2030_n[i] < root2[1]) or (GRE_2030_n[i] <= root1[1] and GRE_2030_n[i] > root2[1]))):
        accuracyGREUnequal += 1

accuracyGPAUnequal *= 100/dataCount
accuracyGREUnequal *= 100/dataCount

print("Unequal Variance GPA Accuracy =",accuracyGPAUnequal)
print("Unequal Variance GRE Accuracy =",accuracyGREUnequal)

#Visualizations
#Equal GPA visualization
for i in range (yesCount):
    marker = 'o'
    color = 'r'
    plt.scatter(GPA_2030_y[i],0,color=color,marker=marker)
for i in range (noCount):
    marker = 'x'
    color = 'b'
    plt.scatter(GPA_2030_n[i],0,color=color,marker=marker)

yBound = np.linspace(-1,1,num=2)
xBound = [bGPAequal,bGPAequal]
plt.plot(xBound,yBound,color='k')

x0 = np.linspace(u0_gpa - 3*np.sqrt(sigEqualGPA), u0_gpa + 3*np.sqrt(sigEqualGPA), 100)
plt.plot(x0, stats.norm.pdf(x0, u0_gpa, sigEqualGPA))

x1 = np.linspace(u1_gpa - 3*np.sqrt(sigEqualGPA), u1_gpa + 3*np.sqrt(sigEqualGPA), 100)
plt.plot(x1, stats.norm.pdf(x1, u1_gpa, sigEqualGPA))

plt.xlabel("GPA")
plt.title("GPA2030 Equal Variances")
plt.show()

for i in range (yesCount):
    marker = 'o'
    color = 'r'
    plt.scatter(GRE_2030_y[i],0,color=color,marker=marker)
for i in range (noCount):
    marker = 'x'
    color = 'b'
    plt.scatter(GRE_2030_n[i],0,color=color,marker=marker)

yBound = np.linspace(-1,1,num=2)
xBound = [bGREequal,bGREequal]
plt.plot(xBound,yBound,color='k')

x0 = np.linspace(u0_gre - 3*np.sqrt(sigEqualGRE), u0_gre + 3*np.sqrt(sigEqualGRE), 100)
plt.plot(x0, stats.norm.pdf(x0, u0_gre, sigEqualGRE))

x1 = np.linspace(u1_gre - 3*np.sqrt(sigEqualGRE), u1_gre + 3*np.sqrt(sigEqualGRE), 100)
plt.plot(x1, stats.norm.pdf(x1, u1_gre, sigEqualGRE))

plt.xlabel("GRE")
plt.title("GRE2030 Equal Variances")
plt.show()

#Unequal variances
for i in range (yesCount):
    marker = 'o'
    color = 'r'
    plt.scatter(GPA_2030_y[i],0,color=color,marker=marker)
for i in range (noCount):
    marker = 'x'
    color = 'b'
    plt.scatter(GPA_2030_n[i],0,color=color,marker=marker)

xBound = np.linspace(2,3,num=25)
yBound = (xBound**2)*a[0]+xBound*b[0]+c[0]
plt.plot(xBound,yBound,color='k')

x0 = np.linspace(u0_gpa - 3*np.sqrt(sigGPAn), u0_gpa + 3*np.sqrt(sigGPAn), 100)
plt.plot(x0, stats.norm.pdf(x0, u0_gpa, sigGPAn))

x1 = np.linspace(u1_gpa - 3*np.sqrt(sigGPAy), u1_gpa + 3*np.sqrt(sigGPAy), 100)
plt.plot(x1, stats.norm.pdf(x1, u1_gpa, sigGPAy))

plt.xlabel("GPA")
plt.title("GPA2030 Unequal Variances")
plt.show()

for i in range (yesCount):
    marker = 'o'
    color = 'r'
    plt.scatter(GRE_2030_y[i],0,color=color,marker=marker)
for i in range (noCount):
    marker = 'x'
    color = 'b'
    plt.scatter(GRE_2030_n[i],0,color=color,marker=marker)

xBound = np.linspace(1.5,2.75,num=25)
yBound = (xBound**2)*a[1]+xBound*b[1]+c[1]
plt.plot(xBound,yBound,color='k')

x0 = np.linspace(u0_gre - 3*np.sqrt(sigGREn), u0_gre + 3*np.sqrt(sigGREn), 100)
plt.plot(x0, stats.norm.pdf(x0, u0_gre, sigGREn))

x1 = np.linspace(u1_gre - 3*np.sqrt(sigGREy), u1_gre + 3*np.sqrt(sigGREy), 100)
plt.plot(x1, stats.norm.pdf(x1, u1_gre, sigGREy))

plt.xlabel("GRE")
plt.title("GRE2030 Unequal Variances")
plt.show()