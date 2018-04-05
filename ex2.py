import numpy as np
from random import shuffle
from math import *
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D


def createExamples(classNum):
    """
    creates 100 examples of class which distributes normally with mean (miu) : 2 * classNum and
    standart deviation (sigma): 1.0
    in our exercise classNum is denoted by 'a'
    :param classNum: number of the class
    :return: list of 100 examples
    """
    return np.random.normal(2 * classNum, 1.0, 100)

def normDistDensity(x, mean, standartDeviation):
    """
    Implementation of the known normal distribution density function.
    :param x: input to function
    :param mean: (miu)
    :param standartDeviation: (sigma)
    :return: output of normal distribution density function.
    """
    return (1.0 / (standartDeviation * sqrt(2 * pi))) * np.exp((-(x - mean) ** 2) / (2 * (standartDeviation**2)))

def sofmax(i, w, xt, b):
    dominator = 0
    for j in range (3):
        dominator += np.exp(w[j] * xt + b[j])
    return np.exp(w[i] * xt + b[i]) / dominator

def training(w, b, eta, s):
    for tag in range(1, 4): #creating examples for
        examples = createExamples(tag)
        for example in examples:
            s.append((example, tag))

    epochs = 20
    for e in range(epochs):
        np.random.shuffle(s)
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

def showTrainingResults(w, b):
    realDist = {}
    trainDist= {}
    for x in range(0, 11):
        realDist[x] = (normDistDensity(x, 2, 1) /
                       (normDistDensity(x, 2, 1) + normDistDensity(x, 4, 1) + normDistDensity(x, 6, 1)))
        trainDist[x] = sofmax(0, w, x, b)
    line1, =plt.plot(realDist.keys(), realDist.values(), "orange", label='Real Distribution')
    line2, =plt.plot(trainDist.keys(), trainDist.values(), "purple", label='Softmax Distribution')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
    plt.show()

def main():
    w = [0, 0, 0]   #classes features matrix
    b = [0, 0, 0]
    eta = 0.3   #learning rate
    s = []  #training set
    training(w, b, eta, s)  # learn w and b vectors according to training set s
    showTrainingResults(w, b)   #draw graphs of the real distribution and the sotmax distribution (according to w, b)

if __name__ == "__main__":
    main()