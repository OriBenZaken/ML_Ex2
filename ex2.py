import numpy as np
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

def softmax(i, w, xt,b):
    """
    return the probability that xt belongs to class i + 1 according to parameters w and b
    :param i: indicates class index
    :param w: classes features matrix
    :param xt: example
    :param b: constants vector
    :return: probability that xt belongs to class i + 1 according to parameters w and b
    """
    dominator = 0
    for j in range (3):
        dominato<r += np.exp(w[j] * xt + b[j])
    return np.exp(w[i] * xt + b[i]) / dominator

def training(w, b, eta, s):
    """
    creates 300 examples, 100 for each of the three classes and go through the training set
    to find the optimal w and b. Training is done in logistic regression method using SGD update rule.
    :param w: classes features matrix
    :param b: constants vector
    :param eta: training rate
    :param s: training set
    :return: no return
    """
    for tag in range(1, 4): # creating examples for
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
                    #  gradient of loss function (softmax) according to w[i] in the example coordinate
                    w_tag = -example + softmax(i, w, example, b) * example
                    #  gradient of loss function (softmax) according to b[i] in the example coordinate
                    b_tag = -1 + softmax(i, w, example, b)
                else:
                    w_tag = softmax(i, w, example, b) * example
                    b_tag = softmax(i, w, example, b)
                w[i] = w[i] - eta * w_tag
                b[i] = b[i] - eta * b_tag

def showTrainingResults(w, b):
    """
    plotting graph of the real distribution and the softmax distribution according to w, b
    for x in range [0, 10]
    :param w: classes features matrix
    :param b: constants vector
    :return:
    """
    realDist = {}
    trainDist= {}
    for x in range(0, 11):
        realDist[x] = (normDistDensity(x, 2, 1) /
                       (normDistDensity(x, 2, 1) + normDistDensity(x, 4, 1) + normDistDensity(x, 6, 1)))
        trainDist[x] = softmax(0, w, x, b)
    line1, =plt.plot(realDist.keys(), realDist.values(), "orange", label='Real Distribution')
    line2, =plt.plot(trainDist.keys(), trainDist.values(), "purple", label='Softmax Distribution')
    # drawing name of the graphs
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
    plt.show()

def main():
    """
    learn w and b and draw the results comparing to the real distribution.
    """
    w = [0, 0, 0]   # classes features matrix - 3 classes (columns), each class got one feature (rows)
    b = [0, 0, 0]   # constants vector
    eta = 0.1   # learning rate
    s = []  # training set
    training(w, b, eta, s)  # learn w and b vectors according to training set s
    showTrainingResults(w, b)   # draw graphs of the real distribution and the softmax distribution (according to w, b)

if __name__ == "__main__":
    main()