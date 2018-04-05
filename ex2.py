import numpy as np
from math import *
from random import shuffle
import matplotlib.pyplot as plt


def createExamples(classNum):
    return np.random.normal(2 * classNum, 1.0, 100);

def normDistDensity(x, mean, standartDeviation):
    return (1.0 / (standartDeviation * sqrt(2 * pi))) * np.exp((-(x - mean) ** 2) / (2 * (standartDeviation**2)))

def sofmax(i, w, xt, b):
    dominator = 0
    for j in range (3):
        dominator += np.exp(w[j] * xt + b[j])
    return np.exp(w[i] * xt + b[i]) / dominator

def main():
    print("Machine Learning Exercise 2:")
    w = [0, 0, 0]   #classes features matrix
    b = [0, 0, 0]
    eta = 0.3   #learning rate
    s = []  #training set


    for tag in range(1, 4): #creating examples for
        count = 0
        examples = createExamples(2 * tag)
        for example in examples:
            s.append((example, tag))

    epochs = 10
    for e in range(epochs):
        for (example, tag) in s:    # example = xt, tag = y
            # updating w and b
            for i in range(3):
                if (i + 1 == tag):
                    w_tag = -example + sofmax(i, w, example, b) * example
                    b_tag = -1 + sofmax(i, w, example, b)
                else:
                    w_tag = sofmax(i, w, example, b) * example
                    b_tag = sofmax(i, w, example, b)
                w[i] = w[i] - eta * w_tag
                b[i] = b[i] - eta * b_tag

    # drawing the graphs
    realDist = {}
    trainDist= {}
    for x in range(0, 11):
        realDist[x] = (normDistDensity(x, 2, 1) /
                       (normDistDensity(x, 2, 1) + normDistDensity(x, 4, 1) + normDistDensity(x, 6, 1)))
        trainDist[x] = sofmax(0, w, x, b) / (sofmax(0, w, x, b) + sofmax(1, w, x, b) + sofmax(2, w, x, b))
    plt.plot(realDist.keys(), realDist.values(), "r-", label='Real')
    plt.plot(trainDist.keys(), trainDist.values(), "b-", label='Real')
    plt.show()


if __name__ == "__main__":
    main()