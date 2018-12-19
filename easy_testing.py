from sklearn import preprocessing
import numpy as np
from models.NeuralNetwork import NeuralNetwork
from models.NaiveBayes import NaiveBayes

X = []
Y = []
# read the training data
count = 0
with open('small_test_set.txt') as f:
    for line in f:
        curr = line.split(',')
        new_curr = [1]
        for item in curr[:len(curr) - 1]:
            new_curr.append(float(item))
        X.append(new_curr)
        Y.append([float(curr[-1])])
        count += 1

X = np.array(X)
X = preprocessing.scale(X) # feature scaling
Y = np.array(Y)

X_train = X[0:int(count * 0.8)]
y_train = Y[0:int(count * 0.8)]
X_test = X[int(count * 0.8) + 1:]
y_test = Y[int(count * 0.8) + 1:]

# Neural Network: 4 neurons in hidden layer
classifier = NeuralNetwork(X_train, y_train, 4)
classifier.train(5000)

# NaiveBayes
classifier2 = NaiveBayes(X_train, y_train)
classifier2.train()

print("Neural Network Accuracy: ", classifier.accuracy(X_test, y_test))
print("Naive Bayes Accuracy:", classifier2.accuracy(X_test, y_test))

