# Name: Ori Ben Zaken
# ID: 311492110

import numpy as np
import pickle

EXAMPLE_VECTOR_SIZE = 784
NUMBER_OF_CLASSES = 10


def read1():
    data = open(r"/home/ori/PycharmProjects/ML_Ex2/data_proccessed").read()
    c = pickle.loads(data)
    train_x = c[0] # contains 5000 vectors of size 784 (28x28) each
    train_y = c[1] # contains 5000 tags for each vector of photo in x_train
    text_x = c[2] # test set
    return [train_x, train_y, text_x]

def train(params, epochs, active_fucntion, learning_rate, train_x, train_y, validtaion_x, validation_y):
    for i in xrange(epochs):
        sum_loss = 0.0
        train_x, train_y = shuffle(train_x, train_y)
        for x, y in zip(train_x, train_y):
            # get vector of probabilities as a result , where index y is the probability that x is classified
            # as tag y
            result_vec = forwardPass(params, active_fucntion, x)
            break
    pass

def forwardPass(params, active_function, x):
    W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
    z1 = np.dot(W1, np.transpose(x))  + b1
    h1 = active_function(z1)
    z2 = np.dot(W2, h1) + b2
    h2 = softmax(W2, h1, b2)
    pass

def shuffle(train_x, train_y):
    # shuffling examples and tags accordingly
    s = np.arange(train_x.shape[0])
    np.random.shuffle(s) # shuffling indices
    train_x = train_x[s]
    train_y = train_y[s]
    return [train_x, train_y]

def softmax(w, h, b):
    result_vec = np.zeros((NUMBER_OF_CLASSES, 1))
    dominator = 0
    for j in range (len(w)):
        dominator += np.exp(np.dot(w[j], h) + b[j])
    for i in xrange(NUMBER_OF_CLASSES):
        result_vec[i] = np.exp(np.dot(w[i], h) + b[i]) / dominator
    return result_vec

# def sigmoid(vec):
#     result = np.zeros((len(vec), 1))
#     for i in range(len(vec)):
#         result[i] = 1 / (1 + np.exp(-vec[i]))
#     return result


def main():
    print ("Machine Learning Ex 3")

    # # loading data from text files
    # train_x = np.loadtxt("train_x") # contains 5000 vectors of size 784 (28x28) each
    # train_y = np.loadtxt("train_y") # contains 5000 tags for each vector of photo in x_train
    # text_x = np.loadtxt("test_x") # test set
    train_x , train_y, test_x = read1()

    print ("Done loading data from text files")

    train_size = len(train_x)
    vaulidaton_size = int(train_size * 0.2)
    validation_x, validation_y = train_x[-vaulidaton_size:, :], train_y[-vaulidaton_size:]
    train_x, train_y = train_x[: -vaulidaton_size, :], train_y[: -vaulidaton_size]

    # Defining hyper-parameters
    h = 300
    w_1 = np.random.rand(h, EXAMPLE_VECTOR_SIZE)
    b_1 = np.random.rand(h)
    w_2 = np.random.rand(NUMBER_OF_CLASSES, h)
    b_2 = np.random.rand(NUMBER_OF_CLASSES)
    sigmoid = lambda x: 1 / (1 + np.exp(-x)) # activation functions
    eta = 0.1
    epochs = 20
    params = {"W1":w_1, "b1":b_1, "W2":w_2, "b2":b_2}
    train(params, epochs, sigmoid, eta, train_x, train_y, validation_x, validation_y)


if __name__ == "__main__":
    main()