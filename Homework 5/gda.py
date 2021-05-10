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

#Mean is simply mean of each group (Confirmed)
u0_gpa = np.mean(GPA_2030_n)
u0_gre = np.mean(GRE_2030_n)
u1_gpa = np.mean(GPA_2030_y)
u1_gre = np.mean(GRE_2030_y)

#Unequal Variances
sigGPAy = np.var(GPA_2030_y)
sigGREy = np.var(GRE_2030_y)
sigGPAn = np.var(GPA_2030_n)
sigGREn = np.var(GRE_2030_n)

#Equal variances is weighted sum of sign and sigy
sigEqualGPA = (noCount/dataCount)*sigGPAn+(yesCount/dataCount)*sigGPAy
sigEqualGRE = (noCount/dataCount)*sigGREn+(yesCount/dataCount)*sigGREy

#Decision Boundaries
#Equal variances
#Following w1x + w0 = 0, we can solve by saying x = -w0/w1 -> Used quadratic equations but updated sigma
w1 = [-2*(u0_gpa/sigEqualGPA-u1_gpa/sigEqualGPA),-2*(u0_gre/sigEqualGRE-u1_gre/sigEqualGRE)]
w0 = [(u0_gpa**2/sigEqualGPA-u1_gpa**2/sigEqualGPA)+np.log((1-py_0)*np.sqrt(sigEqualGPA/sigEqualGPA)/(py_0)),
     (u0_gre**2/sigEqualGRE-u1_gre**2/sigEqualGRE)+np.log((1-py_0)*np.sqrt(sigEqualGRE/sigEqualGRE)/(py_0))]

b = np.zeros(2)

for i in range (2):
    b[i] = -w0[i]/w1[i]

bGPAequal = b[0]
bGREequal = b[1]

print("GPA2030 Statistics")
print("Prior Probability of Non-Attending =",py_0)
print("u_o =",u0_gpa)
print("u_1 =",u1_gpa)
print("Equal Variance =",sigEqualGPA)
print("Variance of Admitted =",sigGPAy)
print("Variance of Non-Admitted =",sigGPAn)
print("Equal Decision Boundary, b =",bGPAequal)
print()
print("__________________")
print("GRE2030 Statistics")
print("Prior Probability of Non-Attending =",py_0)
print("u_o =",u0_gre)
print("u_1 =",u1_gre)
print("Equal Variance =",sigEqualGRE)
print("Variance of Admitted =",sigGREy)
print("Variance of Non-Admitted =",sigGREn)
print("Equal Decision Boundary, b =",bGREequal)

print()
print("______________________________________________")
print("Testing Equal Variance Boundary Conditions")

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


print()
print("______________________________________________")
print("Testing Unequal Variance Boundary Conditions")

a = [1/sigGPAn-1/sigGPAy,1/sigGREn-1/sigGREy]
b = [-2*(u0_gpa/sigGPAn-u1_gpa/sigGPAy),-2*(u0_gre/sigGREn-u1_gre/sigGREy)]
c = [(u0_gpa**2/sigGPAn-u1_gpa**2/sigGPAy)+np.log((1-py_0)*np.sqrt(sigGPAn/sigGPAy)/(py_0)),(u0_gre**2/sigGREn-u1_gre**2/sigGREy)+np.log((1-py_0)*np.sqrt(sigGREn/sigGREy)/(py_0))]

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
#If our x > root2, then it's in class 1 (yes). If our x < root2, then it's in class 0 (no)
accuracyGPAUnequal = 0
accuracyGREUnequal = 0

#Because our roots have 2 outside the range (>4 or <0), we can focus on root 1
for i in range (yesCount):
    if(GPA_2030_y[i] >= root2[0]):
        accuracyGPAUnequal += 1

    if(GRE_2030_y[i] >= root2[1]):
        accuracyGREUnequal += 1

for i in range (noCount):
    if(GPA_2030_n[i] < root2[0]):
        accuracyGPAUnequal += 1
    
    if(GRE_2030_n[i] < root2[1]):
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
plt.plot(xBound,yBound,color='y')

x0 = np.linspace(u0_gpa - 3*np.sqrt(sigEqualGPA), u0_gpa + 3*np.sqrt(sigEqualGPA), 100)
y0 = py_0/np.sqrt(2*np.pi*sigEqualGPA)*np.exp(-(x0-u0_gpa)**2/sigEqualGPA)
plt.plot(x0, y0)

x1 = np.linspace(u1_gpa - 3*np.sqrt(sigEqualGPA), u1_gpa + 3*np.sqrt(sigEqualGPA), 100)
y1 = (1-py_0)/np.sqrt(2*np.pi*sigEqualGPA)*np.exp(-(x1-u1_gpa)**2/sigEqualGPA)
plt.plot(x1, y1)

plt.xlabel("GPA")
plt.title("GPA2030 Equal Variances")
plt.show()

#Equal GRE visualization
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
plt.plot(xBound,yBound,color='y')

x0 = np.linspace(u0_gre - 3*np.sqrt(sigEqualGRE), u0_gre + 3*np.sqrt(sigEqualGRE), 100)
y0 = py_0/np.sqrt(2*np.pi*sigEqualGRE)*np.exp(-(x0-u0_gre)**2/sigEqualGRE)
plt.plot(x0, y0)

x1 = np.linspace(u1_gre - 3*np.sqrt(sigEqualGRE), u1_gre + 3*np.sqrt(sigEqualGRE), 100)
y1 = (1-py_0)/np.sqrt(2*np.pi*sigEqualGRE)*np.exp(-(x1-u1_gre)**2/sigEqualGRE)
plt.plot(x1, y1)

plt.xlabel("GRE")
plt.title("GRE2030 Equal Variances")
plt.show()

#Unequal GPA variances
for i in range (yesCount):
    marker = 'o'
    color = 'r'
    plt.scatter(GPA_2030_y[i],0,color=color,marker=marker)
for i in range (noCount):
    marker = 'x'
    color = 'b'
    plt.scatter(GPA_2030_n[i],0,color=color,marker=marker)

xBound = np.linspace(2.5,3,num=25)
yBound = (xBound**2)*a[0]+xBound*b[0]+c[0]
plt.plot(xBound,yBound,color='k')

yBound = np.linspace(-1,1,num=2)
xBound = [root2[0],root2[0]]
plt.plot(xBound,yBound,color='y')

x0 = np.linspace(u0_gpa - 3*np.sqrt(sigGPAn), u0_gpa + 3*np.sqrt(sigGPAn), 100)
y0 = py_0/np.sqrt(2*np.pi*sigGPAn)*np.exp(-(x0-u0_gpa)**2/sigGPAn)
plt.plot(x0, y0)

x1 = np.linspace(u1_gpa - 3*np.sqrt(sigGPAy), u1_gpa + 3*np.sqrt(sigGPAy), 100)
y1 = (1-py_0)/np.sqrt(2*np.pi*sigGPAy)*np.exp(-(x1-u1_gpa)**2/sigGPAy)
plt.plot(x1, y1)

plt.xlabel("GPA")
plt.title("GPA2030 Unequal Variances")
plt.show()

#Unequal GRE variances
for i in range (yesCount):
    marker = 'o'
    color = 'r'
    plt.scatter(GRE_2030_y[i],0,color=color,marker=marker)
for i in range (noCount):
    marker = 'x'
    color = 'b'
    plt.scatter(GRE_2030_n[i],0,color=color,marker=marker)

xBound = np.linspace(2.5,3,num=25)
yBound = (xBound**2)*a[1]+xBound*b[1]+c[1]
plt.plot(xBound,yBound,color='k')

yBound = np.linspace(-1,1,num=2)
xBound = [root2[1],root2[1]]
plt.plot(xBound,yBound,color='y')

x0 = np.linspace(u0_gre - 3*np.sqrt(sigGREn), u0_gre + 3*np.sqrt(sigGREn), 100)
y0 = py_0/np.sqrt(2*np.pi*sigGREn)*np.exp(-(x0-u0_gre)**2/sigGREn)
plt.plot(x0, y0)

x1 = np.linspace(u1_gre - 3*np.sqrt(sigGREy), u1_gre + 3*np.sqrt(sigGREy), 100)
y1 = (1-py_0)/np.sqrt(2*np.pi*sigGREy)*np.exp(-(x1-u1_gre)**2/sigGREy)
plt.plot(x1, y1)

plt.xlabel("GRE")
plt.title("GRE2030 Unequal Variances")
plt.show()