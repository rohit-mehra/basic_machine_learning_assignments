#!/usr/bin/env python3
"""Assignment 2: UTSA CS 6243/4593 Machine Learning Fall 2017"""

import random
import matplotlib.pyplot as plt
__author__ = "Rohit Mehra"

random.seed(99)


def init_parameters():
    """small random init for w, 0 for b"""
    return {'w': random.uniform(-1, 1) * 0.001, 'b': 0}


def forward_propagation(x, parameters):
    """get prediction"""
    y_hat = parameters['w'] * x + parameters['b']
    return y_hat


def cost(y_hat, y):
    """calc loss"""
    return -(y - y_hat)


def backward_propagation(x, y, y_hat):
    """calc grads"""
    j = cost(y_hat, y) / abs(cost(y_hat, y))
    grads = {'dw': j * x, 'db': j}
    return grads


def update_parameters(parameters, grads, learning_rate):
    w = parameters['w'] - learning_rate * grads['dw']
    b = parameters['b'] - learning_rate * grads['db']

    return {'w': w, 'b': b}


def train(train_x, train_y):
    parameters = init_parameters()
    learning_rate = 0.1
    num_epochs = 10000
    for i in range(num_epochs):

        if i % 1000 == 0:
            # learning_rate decay
            learning_rate /= 10

        # minibatch size = 1
        for x, y in zip(train_x, train_y):
            y_hat = forward_propagation(x, parameters)
            grads = backward_propagation(x, y, y_hat)
            parameters = update_parameters(parameters, grads, learning_rate)

    return parameters


def get_y(x, parameters):
    y = [parameters['w'] * i + parameters['b'] for i in x]
    return y


if __name__ == '__main__':
    train_x = [19.7, 19.1, 18.2, 5.2, 4.3, 9.3, 3.6, 14.8,
               11.9, 9.3, 2.8, 9.9, 15.4, 2.7, 10.6, 16.6, 11.4]

    train_y = [19.7, 19.3, 18.6, 7.9, 4.4, 9.6, 8.0, 15.7,
               15.4, 9.8, 10.3, 11.2, 16.8, 5.1, 12.2, 18.9, 12.2]

    test_x = [18.8,
              15.6,
              17.9]

    # get trained/estimated parameters for linear function
    parameters = train(train_x, train_y)
    print(parameters)
    """{'w': 0.8669155873576103, 'b': 2.8221363164400555}"""

    # predict on test data
    test_y = get_y(test_x, parameters)

    print(test_y)
    """[19.12014935876313, 16.346019479218775, 18.33992533014128]"""

    # plot train data, test data and learnt function
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(train_x, train_y, c='green')
    ax.scatter(test_x, test_y, c='red')

    x = list(range(-1, 21))
    ax.plot(x, get_y(x, parameters), 'b')
    plt.show()
