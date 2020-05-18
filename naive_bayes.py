# Naive Bayes

import pandas as pd
from math import sqrt
from math import exp
from math import pi
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, :].values
trainData, testData = train_test_split(X, test_size = 0.33)

def compareResult(pClass, testClass):
    if (pClass != testClass):
        return 1
    else:
        return 0

def separate_by_class(dataset):
    class0 = []
    class1 = []
    class2 = []
    for i in range(len(dataset)):
        if (dataset[i][-1] == -1):
            class0.append(dataset[i][:-1])
        elif (dataset[i][-1] == 0):
            class1.append(dataset[i][:-1])
        elif (dataset[i][-1] == 1):
            class2.append(dataset[i][:-1])
    sepClasses = [class0, class1, class2]
    return sepClasses

def mean(numbers):
	return sum(numbers)/float(len(numbers))

def stdev(numbers):
	avg = mean(numbers)
	variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
	return sqrt(variance)

def stats_by_class(dataset):
    separated = separate_by_class(dataset)
    statsByClass = {}
    for i in range(len(separated)):
        statsRow = [(mean(column), stdev(column), len(column)) for column in zip(*separated[i])]
        del(statsRow[-1])
        statsByClass[i-1] = statsRow
    return statsByClass

def calculate_probability(x, mean, stdev):
	exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (sqrt(2 * pi) * stdev)) * exponent

def calculate_class_probabilities(model, row):
	total_rows = 100
	probabilities = {}
	for classValue, classItems in model.items():
		probabilities[classValue] = model[classValue][0][2]/float(total_rows)
		for i in range(len(classItems)):
			probabilities[classValue] *= calculate_probability(row[i], classItems[i][0], classItems[i][1])
	return probabilities

def classify(model, row):
    probabilities = calculate_class_probabilities(model, row)
    prob = 0
    for class_value, probability in probabilities.items():
        if (prob < probability):
            prediction = class_value
            prob = probability
    return prediction

def main():   
    model = stats_by_class(trainData)
    classVec = []
    errors = 0
    for i in range(len(testData)):
         pClass = classify(model, testData[i])
         classVec.append(pClass)
         errors += compareResult(pClass, testData[i][-1])
    
    print(classVec)
    print(errors)
    
if '__main__' == main():
    main()

