import pandas as pd
from math import sqrt
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, :].values
train, test = train_test_split(X, test_size = 0.33)

def compare_result(pClass, testClass):
    if (pClass != testClass):
        return 1
    else:
        return 0

def eucledian_distance(testP, trainP):
    suma = 0
    for i in range(len(trainP)):
        suma += (trainP[i] - testP[i])**2
    return sqrt(suma)

#very very bad function, only works with this dataset
def classify(array):
    counter = [0] * 3
    for i in array:
        if (i[1] == -1):
            counter[0] += 1
        elif (i[1] == 0):
            counter[1] += 1
        elif (i[1] == 1):
            counter[2] += 1
    maxValue = max(counter)
    return counter.index(maxValue) - 1

def main():
    
    K = 3
    classVec = []
    errors = 0
    
    
    for i in range(len(test)):
        cNeighbors = []
        for j in range(len(train)):
            dist = eucledian_distance(test[i], train[j])
            cNeighbors.append([dist, train[j][-1]])
        
        cNeighbors.sort()
        
        pClass = classify(cNeighbors[:K])
        
        classVec.append(pClass)
        
        errors += compare_result(pClass, test[i][-1])
        
    print(classVec)
    print(errors)

if '__main__' == main():
    main()