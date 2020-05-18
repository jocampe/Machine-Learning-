import numpy as np
import random as rand
import math
import matplotlib.pyplot as plt

def costFuntion(x):
    return schwefel(x)

def main():
    MaxGen = 2000
    Np = 50 
    Dim = 10
    Cr = 0.9 
    F = 0.5 
    xHigh = 500 #borders
    xLow = -500
    avgVec = [0] * MaxGen   #init avg plot 
    for k in range(3):
        Gen = 0
        cVal = [4209000]  #lazyness
    
        X = np.zeros((Np, Dim))
        U = np.zeros((Np, Dim))

        #initialization
        for i in range(Np):
            for j in range(Dim):
                X[i][j] = xLow + rand.random()*(xHigh - xLow)
    
        #3.
        while(Gen < MaxGen):
            #4.
            for i in range(Np):
                #4.1 
                r1, r2, r3 = i, i, i
                while(r1 == i):
                    r1 = getRandom(Np)
                while(r2 == r1 or r2 == i):
                    r2 = getRandom(Np)
                while(r3 == r2 or r3 == r1 or r3 == i):
                    r3 = getRandom(Np)
                    
                #4.2
                jrand = getRandom(Dim)
                
                #4.3
                for j in range(Dim):
                    if (rand.random() <= Cr or j == jrand):
                        U[i][j] = X[r3][j] + F * (X[r1][j] - X[r2][j])
                    else:
                        U[i][j] = X[i][j]
            
                #5
                if costFuntion(U[i]) <= costFuntion(X[i]):
                    X[i] = U[i]
                
                fixBorders(X[i], Dim, xHigh, xLow)
            
            #filling the CF value array
            cVal.append(getCF(X, cVal[-1]))
            
            Gen += 1
            
        plt.plot(cVal[1:])
        #Update the average plot
        for i in range(len(cVal)-1):
            avgVec[i] = (avgVec[i]*k + cVal[i+1])/(k+1)
    plt.title('Evolution of 30 convergence lines for 200gen')
    plt.ylabel('CF Value')
    plt.xlabel('Generations')
    plt.show()
    
    plt.plot(avgVec)
    plt.title('Average evolution')
    plt.ylabel('CF Value')
    plt.xlabel('Generations')
    plt.show()
    

def deJong1(array):
    return sum(array**2)

def deJong2(x):
    total = 0
    for i in range(len(x)-2):
        total += 100 * ((x[i+1] - (x[i]**2))**2) + ((1 - (x[i]**2))**2)
    return total

def schwefel(x):
    total = 0
    for n in x:
        total += -n*math.sin(math.sqrt(math.fabs(n)))
    return total + len(x)*418.982887

def getRandom(mul):
    return math.floor(rand.random() * mul)

def getCF(x, prevCF):
    a = prevCF
    for i in range(50):
        b = costFuntion(x[i])
        if (a > b):
            a = b
    return a

def fixBorders(X, dim, xHigh, xLow):
    for i in range(dim):
        if X[i] > xHigh:
            X[i] = xLow + rand.random()*(xHigh - xLow)
        if X[i] < xLow:
            X[i] = xLow + rand.random()*(xHigh - xLow)

if '__main__' == main():
    main()