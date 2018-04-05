import numpy as np
from math import *
from random import shuffle
import matplotlib.pyplot as plt


def createExamples(classNum):
    return np.random.normal(2 * classNum, 1.0, 100);

def normDistDensity(x, mean, standartDeviation):
    return (1.0 / (standartDeviation * sqrt(2 * pi))) * np.exp((-(x - mean) ** 2) / (2 * (standartDeviation**2)))

def main():
    print("Machine Learning Exercise 2:")
    w = [0, 0, 0]   #classes features matrix
    b = [0, 0, 0]
    eta = 0.1   #learning rate
    s = []  #training set


    for tag in range(1, 4): #creating examples for
        count = 0
        examples = createExamples(2 * tag)
        for example in examples:
            s.append((example, tag))

    epochs = 10
    #for e in range(epochs):
    realDist = {}
    for x in range(0, 11):
        realDist[x] = (normDistDensity(x, 2, 1) /
                       (normDistDensity(x, 2, 1) + normDistDensity(x, 4, 1) + normDistDensity(x, 6, 1)))
    plt.plot(realDist.keys(), realDist.values(), "r-", label='Real')
    plt.show()


if __name__ == "__main__":
    main()