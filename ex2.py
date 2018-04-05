import numpy as np
from random import shuffle

def createExamples(classNum):
    return np.random.normal(2 * classNum, 1.0, 100);

def main():
    print("Machine Learning Exercise 2:")
    w = [0, 0, 0]   #classes features matrix
    b = [0, 0, 0]
    eta = 0.1   #learning rate
    s = []  #training set
    for tag in range(1, 4):
        count = 0
        examples = createExamples(2 * tag)
        for example in examples:
            s.append((example, tag))
    shuffle(s)


if __name__ == "__main__":
    main()