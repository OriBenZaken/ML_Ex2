# Name: Ori Ben Zaken
# ID: 311492110

import numpy as np
import pickle

EXAMPLE_VECTOR_SIZE = 784
NUMBER_OF_CLASSES = 10

sigmoid = lambda x: np.divide(1, (1 + np.exp(-x)))  # activation functions


def read1():
    data = open(r"/home/ori/PycharmProjects/ML_Ex2/data_proccessed").read()
    c = pickle.loads(data)
    train_x = c[0] # contains 5000 vectors of size 784 (28x28) each
    train_y = c[1] # contains 5000 tags for each vector of photo in x_train
    text_x = c[2] # test set
    return [train_x, train_y, text_x]

def train(params, epochs, active_fucntion, learning_rate, train_x, train_y, validtaion_x, validation_y):
    for i in xrange(epochs):
        print ("Epoch no. {0}".format(i + 1))
        sum_loss = 0.0
        train_x, train_y = shuffle(train_x, train_y)
        for x, y in zip(train_x, train_y):
            x = np.reshape(x, (1, EXAMPLE_VECTOR_SIZE))
            # get vector of probabilities as a result , where index y is the probability that x is classified
            # as tag y
            fprop_cache = forwardPass(params, active_fucntion, x, y)
            y_predict = fprop_cache["y_predict"]
            loss = loss_function(y, y_predict)
            sum_loss += loss
            backPropagation(fprop_cache)


    pass

def forwardPass(params, active_function, x, y):
    W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
    z1 = np.dot(W1, np.transpose(x))  + b1
    h1 = active_function(z1)
    z2 = np.dot(W2, h1) + b2
    y_predict = softmax(W2, h1, b2)
    ret = {'x': x, 'y':y, 'z1': z1, 'h1': h1, 'z2': z2, 'y_predict': y_predict}
    for key in params:
        ret[key] = params[key]
    return ret

def bprop(f_prop_cache):
  # Follows procedure given in notes
  x, y, z1, h1, z2, y_predict = [fprop_cache[key] for key in ('x', 'y', 'z1', 'h1', 'z2', 'y_predict')]
  one_hot_vec = np.zeros((NUMBER_OF_CLASSES, 1))
  one_hot_vec[int(y)] = 1
  dz2 = (y_predict - one_hot_vec)                                #  dL/dz2
  dW2 = np.dot(dz2, h1.T)                       #  dL/dz2 * dz2/dw2
  db2 = dz2                                     #  dL/dz2 * dz2/db2
  dz1 = np.dot(fprop_cache['W2'].T,
    (y_predict - one_hot_vec)) * sigmoid(z1) * (1-sigmoid(z1))   #  dL/dz2 * dz2/dh1 * dh1/dz1
  dW1 = np.dot(dz1, x.T)                        #  dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/dw1
  db1 = dz1                                     #  dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/db1
  return {'b1': db1, 'W1': dW1, 'b2': db2, 'W2': dW2}

# calculate gradients
def backPropagation(fprop_cache):
    x, y, z1, h1, z2, y_predict = [fprop_cache[key] for key in ('x', 'y', 'z1', 'h1', 'z2', 'y_predict')]
    one_hot_vec = np.zeros((NUMBER_OF_CLASSES, 1))
    one_hot_vec[int(y)] = 1
    dz2 = (y_predict - one_hot_vec)  # dL/dz2
    dW2 = np.dot(dz2, h1.T)  # dL/dz2 * dz2/dw2
    db2 = dz2  # dL/dz2 * dz2/db2
    dz1 = np.dot(fprop_cache['W2'].T,
                 (y_predict - one_hot_vec)) * sigmoid(z1) * (1 - sigmoid(z1))  # dL/dz2 * dz2/dh1 * dh1/dz1
    dW1 = np.dot(dz1, x)  # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/dw1
    db1 = dz1  # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/db1
    return {'b1': db1, 'W1': dW1, 'b2': db2, 'W2': dW2}

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

def loss_function(y, y_predict):
    return -np.log(y_predict[int(y)])

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

    # normalize examples
    train_x = train_x / 255
    test_x = test_x / 255

    train_size = len(train_x)
    vaulidaton_size = int(train_size * 0.2)
    validation_x, validation_y = train_x[-vaulidaton_size:, :], train_y[-vaulidaton_size:]
    train_x, train_y = train_x[: -vaulidaton_size, :], train_y[: -vaulidaton_size]

    # Defining hyper-parameters
    h = 30
    w_1 = np.random.rand(h, EXAMPLE_VECTOR_SIZE)
    b_1 = np.random.rand(h, 1)
    w_2 = np.random.rand(NUMBER_OF_CLASSES, h)
    b_2 = np.random.rand(NUMBER_OF_CLASSES, 1)
    eta = 0.1
    epochs = 20
    params = {"W1":w_1, "b1":b_1, "W2":w_2, "b2":b_2}
    train(params, epochs, sigmoid, eta, train_x, train_y, validation_x, validation_y)


if __name__ == "__main__":
    main()