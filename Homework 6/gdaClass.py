import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

#Load in the data and disperse respectively
grad_2030 = pd.read_csv("UCLA_EE_grad_2030.csv",header=None)

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

#Divide into our Accepted/NonAccepted Groups
data_y = np.zeros((yesCount,2))
data_n = np.zeros((noCount,2))

for i in range (yesCount):
    data_y[i][0] = GPA_2030_y[i]
    data_y[i][1] = GRE_2030_y[i]

for i in range (noCount):
    data_n[i][0] = GPA_2030_n[i]
    data_n[i][1] = GRE_2030_n[i]

#Calculate parameters
#Priors based on Bernoulli
py_0 = noCount/dataCount

#Mean is simply mean of each group (Confirmed)
u0 = np.mean(data_n,axis=0)
u1 = np.mean(data_y,axis=0)

#We want our covariance matrix to be 2x2, so we can do a (2x100)x(100x2)
cov_n = np.cov(data_n.T)
cov_y = np.cov(data_y.T)

sig = (noCount/dataCount)*cov_n+(yesCount/dataCount)*cov_y
sigInv = np.linalg.inv(sig)

print("2030 GDA Statistics")
print("Prior Probability of Non-Attending =",py_0)
print("u_o =",u0)
print("u_1 =",u1)
print("Covariance Matrix =", sig)
print("Inverse Covariance Matrix =", sigInv)
print()

#Calculate w and b
print("Calculating Boundaries")

w = (u0.T.dot(sigInv)-u1.T.dot(sigInv)).T
b = 0.5*(u1.T.dot(sigInv.dot(u1))-u0.T.dot(sigInv.dot(u0)))+py_0-(1-py_0)

print("w =",w)
print("b =",b)

#Plot
#Equal GPA visualization
for i in range (yesCount):
    marker = 'o'
    color = 'r'
    plt.scatter(data_y[i][0],data_y[i][1],color=color,marker=marker)
for i in range (noCount):
    marker = 'x'
    color = 'b'
    plt.scatter(data_n[i][0],data_n[i][1],color=color,marker=marker)

x1 = np.linspace(1.5,4,num=25)
numer = w[0]*x1 + b
x2 = -numer/w[1]
plt.plot(x1,x2,'-k')
plt.xlabel("GPA")
plt.ylabel("Normalized GRE Score")
plt.title("UCLA EE Graduate 2030 Plot")
plt.show()

#Create functions to easily calculate our distribution
def class0(data):
    pred = py_0*np.exp(-0.5*(data-u0).T.dot(sigInv.dot(data-u0)))
    return pred

def class1(data):
    pred = (1-py_0)*np.exp(-0.5*(data-u1).T.dot(sigInv.dot(data-u1)))
    return pred

#Probability Equality
#Create GPA/GRE MeshGrids from 0-4
axes = np.linspace(0,4)
meshLength = len(axes)
GPA,GRE = np.meshgrid(axes,axes)
Z = np.zeros_like(GPA)

for i in range (meshLength):
    for j in range (meshLength):
        data = np.array([GPA[i][j],GRE[i][j]])
        
        prediction = class1(data)-class0(data)
        
        Z[i][j] = np.sign(prediction)

plt.contourf(GPA,GRE,Z)
x1 = np.linspace(1.75,4,num=25)
numer = w[0]*x1 + b
x2 = -numer/w[1]
plt.plot(x1,x2,'-k')
plt.xlabel("GPA")
plt.ylabel("Normalized GRE Score")
plt.title("UCLA EE Graduate 2030 Plot")
for i in range (yesCount):
    marker = 'o'
    color = 'r'
    plt.scatter(data_y[i][0],data_y[i][1],color=color,marker=marker)
for i in range (noCount):
    marker = 'x'
    color = 'b'
    plt.scatter(data_n[i][0],data_n[i][1],color=color,marker=marker)
plt.show()