
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from google.colab import files
uploaded = files.upload()

class Perceptron(object):

    #Constructor
    def __init__(self, lr=0.01, passIter=50, rndState=1):
        self.lr = lr
        self.passIter = passIter
        self.rndState = rndState

    #Fit training data
    def fit(self, X, y):
      
        rn = np.random.RandomState(self.rndState)
        self.nextWeight = rn.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errorList = []

        for _ in range(self.passIter):
            errorNumber = 0
            for xi, target in zip(X, y):
                update = self.lr * (target - self.predict(xi))
                self.nextWeight[1:] += update * xi
                self.nextWeight[0] += update
                errorNumber += int(update != 0.0)
            self.errorList.append(errorNumber)
        return self

    #Calculate net input
    def net_input(self, X): 
        return np.dot(X, self.nextWeight[1:]) + self.nextWeight[0]

    #To return label after unit
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

#v1 = np.array([1, 2, 3])
#v2 = 0.5 * v1
#np.arccos(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

df = pd.read_csv('irismodified.csv', header=None, encoding='utf-8')
df.tail()

# Select setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# Find petal length and sepal length
X = df.iloc[0:100, [0, 2]].values

# Plotting the dataset
plt.scatter(X[:50, 0], X[:50, 1],
            color='green', marker='x', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='orange', marker='o', label='versicolor')

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='lower right') 
plt.show()

# Training the perceptrons
pr = Perceptron(lr=0.1, passIter=10)

pr.fit(X, y)

plt.plot(range(1, len(pr.errorList) + 1), pr.errorList, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()

# A function for plotting decision regions
def plot_decision_regions(X, y, classify, resolution=0.02):

    #  markers and colors
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('green', 'yellow', 'lightgreen', 'gray', 'cyan')
    colorMap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classify.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, colorMap=colorMap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=cl, edgecolor='black')
 
plot_decision_regions(X, y, classify=pr)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
 
plt.show()

class AdalineGD(object):
 
    def __init__(self, lr=0.01, passIter=50, rndState=1):
        self.lr = lr
        self.passIter = passIter
        self.rndState = rndState

    def fit(self, X, y): 
        rgen = np.random.RandomState(self.rndState)
        self.nextWeight = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.passIter):
            net_input = self.net_input(X) 
            output = self.activation(net_input)
            errors = (y - output)
            self.nextWeight[1:] += self.lr * X.T.dot(errors)
            self.nextWeight[0] += self.lr * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        #Calculate net input
        return np.dot(X, self.nextWeight[1:]) + self.nextWeight[0]

    def activation(self, X):
        #Compute linear activation
        return X

    def predict(self, X):
        #Return class label after unit step
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
 
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

ada1 = AdalineGD(passIter=10, lr=0.01).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')

ada2 = AdalineGD(passIter=10, lr=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')
 
plt.show()

X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
 
ada_gd = AdalineGD(passIter=15, lr=0.01)
ada_gd.fit(X_std, y)

plot_decision_regions(X_std, y, classify=ada_gd)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout() 
plt.show()

plt.plot(range(1, len(ada_gd.cost_) + 1), ada_gd.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')

plt.tight_layout() 
plt.show()
 
#Large scale machine learning and stochastic gradient descent
class AdalineSGD(object): 
    def __init__(self, lr=0.01, passIter=10, shuffle=True, rndState=None):
        self.lr = lr
        self.passIter = passIter
        self.w_initialized = False
        self.shuffle = shuffle
        self.rndState = rndState
        
    def fit(self, X, y): 
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.passIter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        #Fit training data without reinitializing the weights
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        #Shuffle training data
        r = self.rgen.permutation(len(y))
        return X[r], y[r]
    
    def _initialize_weights(self, m):
        #Initialize weights to small random numbers
        self.rgen = np.random.RandomState(self.rndState)
        self.nextWeight = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)
        self.w_initialized = True
        
    def _update_weights(self, xi, target):
        #Apply Adaline learning rule to update the weights
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.nextWeight[1:] += self.lr * xi.dot(error)
        self.nextWeight[0] += self.lr * error
        cost = 0.5 * error**2
        return cost
    
    def net_input(self, X):
        #Calculate net input
        return np.dot(X, self.nextWeight[1:]) + self.nextWeight[0]

    def activation(self, X):
        #Compute linear activation
        return X

    def predict(self, X):
        #Return class label after unit step
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
 
ada_sgd = AdalineSGD(passIter=15, lr=0.01, rndState=1)
ada_sgd.fit(X_std, y)

plot_decision_regions(X_std, y, classify=ada_sgd)
plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')

plt.tight_layout() 
plt.show()

plt.plot(range(1, len(ada_sgd.cost_) + 1), ada_sgd.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')

plt.tight_layout() 
plt.show()

ada_sgd.partial_fit(X_std[0, :], y[0])