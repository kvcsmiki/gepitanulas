from numpy import size, zeros, round
from sigmoid import sigmoid

# PREDICT Predict whether the label is 0 or 1 using learned logistic
# regression parameters theta
#   p = PREDICT(theta, X) computes the predictions for X using a
#   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

def predict(theta, X):
    m = size(X, 0)  # Number of training examples

    # You need to return the following variables correctly
    p = zeros(m, dtype=float)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the following code to make predictions using
    #               your learned logistic regression parameters.
    #               You should set p to a vector of 0's and 1's
    #

    p = round(sigmoid(X @ theta))

    return p
    # =========================================================================
