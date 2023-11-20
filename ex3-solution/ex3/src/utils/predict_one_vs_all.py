import numpy as np
from .sigmoid import sigmoid


def predict_one_vs_all(all_theta, X):
    """
    Predict the label for a trained one-vs-all classifier.

    :param all_theta: matrix containing all thetas for each class. The i_th row contains the thetas for class i
    :param X: the training set

    :return: p vector of predictions for each X element
    """

    m = np.size(X, 0)
    num_labels = np.size(all_theta, 0)

    """
    ================================================ YOUR CODE HERE ====================================================
    Instructions: Complete the following code to make predictions using your learned logistic regression
                  parameters (one-vs-all). You should set p to a vector of predictions, p[i] in range(1, num_labels).

    Hint: you can use `np.argmax` to get the index of a given element in an array.
          Example usage when dealing with a vector
          ```
          p = np.argmax(output)  #  returns the index of the output vector's maximum element
          ```
    """

    p = []
    x = np.c_[np.ones((m, 1)), X]

    # for i in range(0, m):
    #     tmp = []
    #
    #     for j in range(0, num_labels):
    #         p_ij = np.matmul(x[i], all_theta[j].T)
    #         tmp.append(p_ij)
    #
    #     #  index 0 means it's in class 1
    #     p.append(np.argmax(tmp) + 1)

    tmp = np.zeros((m,num_labels))
    for i in range(1,num_labels):
        tmp[:,i-1] = sigmoid(np.dot(x,all_theta[i-1,:].T))

    p = tmp.argmax(axis=-1)+1
    return p

