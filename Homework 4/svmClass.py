import numpy as np
import cvxpy as cp
import pandas as pd
import matplotlib.pyplot as plt

#Load in the data and disperse respectively
grad_2031 = pd.read_csv("UCLA_EE_grad_2031_3.csv",header=None)

GPA = grad_2031[0].tolist()
GRE = grad_2031[1].tolist()
labels = grad_2031[2].tolist()

dataCount = len(labels)

#2031 Data Plotted
for i in range (dataCount):
    if(labels[i] == 1):
        marker = 'o'
        color = 'r'
    else:
        marker = 'x'
        color = 'b'
    plt.scatter(GPA[i],GRE[i],marker=marker,color=color)

plt.xlabel("GPA")
plt.ylabel("Normalized GRE Score")
plt.title("UCLA EE Graduate 2031 Plot")

#Problem data
n = 2 #Two features (X1 and X2)
m = 100

#Construct the problem
w = cp.Variable((n))
b = cp.Variable()
objective = cp.Minimize(1/2*cp.sum_squares(w))

#Constraints must all be satisfied
#We need: labels*(w.T.dot([GPA, GRE] + b)) >= 1
x = np.column_stack((GPA,GRE))

constraints = [labels[i] * (w.T @ x[i].T + b) >= 1 for i in range (m)]

prob = cp.Problem(objective, constraints)

result = prob.solve()

wNew = w.value
bNew = b.value

x1 = np.linspace(2.5,4,num=16)
numer = wNew[0]*x1 + bNew
x2 = -numer/wNew[1]
plt.plot(x1,x2,'-k')
plt.show()

print("Part B")
print("w =",w.value)
print("b =",b.value)
print()

#Part c
#Problem data
n = 2 #Two features (X1 and X2)
m = 100
x = np.column_stack((GPA,GRE))
errorTerm = 1e-2*np.eye(m)

#Construct P matrix first
P = np.zeros((m,m))
for i in range (m):
    for j in range (m):
        P[i][j] = labels[i]*labels[j]*x[i].T.dot(x[j])
P += errorTerm
P *= 1/2

#Construct the problem
a = cp.Variable((m))
objective = cp.Maximize(cp.sum(a)-cp.quad_form(a,P))

#Constraints
constraints = [cp.sum(a @ labels) == 0]
constraints += [a[i] >= 0 for i in range (m)]

prob = cp.Problem(objective, constraints)

result = prob.solve()

aNew = a.value

print("Part C")
print("a =",aNew)