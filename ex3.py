# Name: Ori Ben Zaken
# ID: 311492110

import numpy as np
import pickle

# defines - hyper parameters.
EXAMPLE_VECTOR_SIZE = 784
NUMBER_OF_CLASSES = 10
HIDDEN_LAYER_SIZE = 100
EPOCHS = 35
LEARNING_RATE = 0.005

# activation function
sigmoid = lambda x: np.divide(1, (1 + np.exp(-x)))  # activation functions

def train(params, epochs, active_function, learning_rate, train_x, train_y, validtaion_x, validation_y):
    """
    trains the model with SGD method
    :param params: params for all the layers in the neural network
    :param epochs: number of epochs
    :param active_function: activation function (must be not linear)
    :param learning_rate: learning rate
    :param train_x: train set
    :param train_y: labels of the training set
    :param validtaion_x: validation set
    :param validation_y: labels of the validation set
    :return: params of the neural network layers after learning
    """
    for i in xrange(epochs):
        print ("Epoch no. {0}".format(i + 1))
        sum_loss = 0.0
        # shuffle train examples - helps the model to learn (won't just remember order)
        train_x, train_y = shuffle(train_x, train_y)
        for x, y in zip(train_x, train_y):
            x = np.reshape(x, (1, EXAMPLE_VECTOR_SIZE))
            # get vector of probabilities as a result , where index y is the probability that x is classified
            # as tag y
            fprop_cache = forwardPass(params, active_function, x, y)
            y_predict = fprop_cache["y_predict"]
            index = y_predict.argmax(axis=0)
            loss = loss_function(y, y_predict)
            sum_loss += loss
            # gradients for each parameter
            gradients = backPropagation(fprop_cache)
            # update the parameters
            params = update_weights_sgd(params, gradients, learning_rate)

        val_loss, acc = predict_on_validation(params, active_function, validtaion_x, validation_y)
        print ("Test avg loss: {0}".format(sum_loss / len(train_x)))
        print("Validation set results: avg loss: {0}, acc: {1}%".format(val_loss, acc * 100))
    return params

def forwardPass(params, active_function, x, y):
    """
    move example x through the layers of the neural network
    :param params: neural network parameters
    :param active_function: activation function
    :param x: example
    :param y: true label
    :return: softmax vector (classification probabilities vector)
    """
    W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
    z1 = np.dot(W1, np.transpose(x))  + b1
    h1 = active_function(z1)
    z2 = np.dot(W2, h1) + b2
    y_predict = softmax(W2, h1, b2)
    ret = {'x': x, 'y':y, 'z1': z1, 'h1': h1, 'z2': z2, 'y_predict': y_predict}
    for key in params:
        ret[key] = params[key]
    return ret


def backPropagation(fprop_cache):
    """
    calculate gradients
    :param fprop_cache: parameters and mid calculations from the forward pass
    :return: model's parameters gradients
    """
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

def update_weights_sgd(params, gradients, learning_rate):
    """
    updating parameters of the model
    :param params: model parameters
    :param gradients: gradients of the model parameters
    :param learning_rate: learning rate
    :return: updated parameters
    """
    for one_param in params:
        params[one_param] += -learning_rate * gradients[one_param]
    return params

def predict_on_validation(params, active_function, validation_x, validation_y):
    """
    validation stage. calculating average loss and accuracy with no learning, on the validation set
    :param params: model parameters
    :param active_function: activation function (most be not linear)
    :param validation_x: validation set
    :param validation_y: labels of the validation set
    :return: list: [average loss, accuracy]
    """
    sum_loss = 0.0
    good = 0.0

    for x, y in zip(validation_x, validation_y):
        x = np.reshape(x, (1, EXAMPLE_VECTOR_SIZE))
        # get vector of probabilities as a result , where index y is the probability that x is classified
        # as tag y
        fprop_cache = forwardPass(params, active_function, x, y)
        y_predict = fprop_cache["y_predict"]
        loss = loss_function(y, y_predict)
        sum_loss += loss
        max_index_array = y_predict.argmax(axis=0)
        if int(y) == max_index_array[0]:
            good += 1

    acc = good / len(validation_x)
    avg_loss = sum_loss / len(validation_x)

    return avg_loss, acc

def shuffle(train_x, train_y):
    """
    shuffling examples and tags accordingly
    :param train_x: training set
    :param train_y: labels of the training set
    :return: shuffled train_x, train_y
    """
    s = np.arange(train_x.shape[0])
    np.random.shuffle(s) # shuffling indices
    train_x = train_x[s]
    train_y = train_y[s]
    return [train_x, train_y]

def softmax(w, h, b):
    """
    softmax function
    :param w: w parameter of the last layer in the model
    :param h: input vector
    :param b: b parameter of the last layer in the model
    :return: elemnt wise softmax vector
    """
    result_vec = np.zeros((NUMBER_OF_CLASSES, 1))
    dominator = 0
    for j in range (len(w)):
        dominator += np.exp(np.dot(w[j], h) + b[j])
    for i in xrange(NUMBER_OF_CLASSES):
        result_vec[i] = np.exp(np.dot(w[i], h) + b[i]) / dominator
    return result_vec

def loss_function(y, y_predict):
    """
    loss function
    :param y: real label
    :param y_predict: prediction
    :return: loss function output
    """
    return -np.log(y_predict[int(y)])

def test(test_x, params):
    """
    write the model predictions on the test set to test.pred file
    :param test_x: test set
    :param params: model parameters
    :return: None
    """
    with open("test.pred", 'w') as test_pred_file:
        prediction_list = []
        for x in test_x:
            x = np.reshape(x, (1, EXAMPLE_VECTOR_SIZE))
            fprop_cache = forwardPass(params, sigmoid, x, None)
            y_predict = fprop_cache["y_predict"]
            index = y_predict.argmax(axis=0)
            prediction_list.append(str(index[0]))
        test_pred_file.write('\n'.join(prediction_list))


def main():
    """
    builds a neural network with one hidden layer model on FASHION-MNIST data set
    :return:
    """
    print ("Machine Learning Ex 3")

    # loading data from text files
    train_x = np.loadtxt("train_x") # contains 5000 vectors of size 784 (28x28) each
    train_y = np.loadtxt("train_y") # contains 5000 tags for each vector of photo in x_train
    test_x = np.loadtxt("test_x") # test set

    print ("Done loading data from text files")

    # normalize examples
    train_x = train_x / 255
    test_x = test_x / 255

    train_size = len(train_x)
    validation_size = int(train_size * 0.2)
    validation_x, validation_y = train_x[-validation_size:, :], train_y[-validation_size:]
    train_x, train_y = train_x[: -validation_size, :], train_y[: -validation_size]

    # Defining hyper-parameters
    h = HIDDEN_LAYER_SIZE
    w_1 = np.random.uniform(-0.08, 0.08, [h, EXAMPLE_VECTOR_SIZE])
    b_1 = np.random.uniform(-0.08, 0.08, [h, 1])
    w_2 = np.random.uniform(-0.08, 0.08, [NUMBER_OF_CLASSES, h])
    b_2 = np.random.uniform(-0.08, 0.08,[NUMBER_OF_CLASSES, 1])
    eta = LEARNING_RATE
    epochs = EPOCHS
    params = {"W1":w_1, "b1":b_1, "W2":w_2, "b2":b_2}

    params = train(params, epochs, sigmoid, eta, train_x, train_y, validation_x, validation_y)

    test(test_x, params)


if __name__ == "__main__":
    main()