import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

soma = 0
tSize = 0.2
activation_functionH = 'tanh'
activation_functionO = 'softmax'

dataset = pd.read_csv('iris.data')
X = dataset.iloc[:, :-1].values
y1 = dataset.iloc[:, [4]].values

encoder = OneHotEncoder(sparse=False, categories='auto')
y3 = encoder.fit_transform(y1)

X_train, X_test, y_train, y_test = train_test_split(X, y3, test_size = tSize, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = Sequential()
classifier.add(Dense(output_dim = 15, input_shape=(4,), activation=activation_functionH))
classifier.add(Dense(output_dim = 15, activation=activation_functionH))
classifier.add(Dense(output_dim = 3, activation=activation_functionO, name='output'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

history = classifier.fit(X_train, y_train, batch_size = 10, epochs = 200)

y_pred = classifier.predict(X_test)

y_pred1 = np.argmax(y_pred, axis=1)
y_test1 = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_test1, y_pred1)

for array in y_pred:
    soma += max(array)
testAverage = soma/(tSize*len(X))

plt.plot(history.history['accuracy'])
plt.title('sigmoid hidden / Softmax output')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

objects = ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica')
y_pos = np.arange(len(objects))
performance = [cm[0][0],cm[1][1],cm[2][2]]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Occurencies')
plt.title('Distribution via classes')
plt.show()
