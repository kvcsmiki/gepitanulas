#!/usr/bin/env python3
from numpy import zeros, mean, loadtxt, array, concatenate, shape, ones
import os
import matplotlib.pyplot as plt
from plotData import plotData
from sigmoid import sigmoid
from costFunction import costFunction
from predict import predict
from scipy.optimize import minimize
from plotDecisionBoundary import plotDecisionBoundary

# Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the logistic
#  regression exercise. You will need to complete the following functions
#  in this exericse:
#
#     sigmoid.py
#     costFunction.py
#     predict.py
#     costFunctionReg.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#


def ex2():
    # Initialization
    os.system("cls" if os.name == "nt" else "clear")

    # Load Data
    data = loadtxt("ex2data1.txt", delimiter=",")
    #  The first two columns contains the exam scores and the third column
    #  contains the label.
    X = data[:, 0:2]
    y = data[:, 2]

    # ==================== Part 1: Plotting ====================
    #  We start the exercise by first plotting the data to understand the
    #  the problem we are working with.

    print("Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.")

    plotData(X, y)

    # Put some labels
    # Labels and Legend
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")

    # Specified in plot order
    plt.plot([], [], "bo", label="Admitted")
    plt.plot([], [], "r*", label="Not admitted")
    plt.legend()
    plt.show()

    input("Program paused. Press enter to continue.")

    # ============ Part 2: Compute Cost and Gradient ============
    #  In this part of the exercise, you will implement the cost and gradient
    #  for logistic regression. You neeed to complete the code in
    #  costFunction.py

    #  Setup the data matrix appropriately, and add ones for the intercept term
    m, n = shape(X)

    # Add intercept term to x and X_test
    X = concatenate((ones((m, 1)), X), axis=1)

    # Initialize fitting parameters
    initial_theta = zeros(n + 1)

    # Compute and display initial cost and gradient
    cost, grad = costFunction(initial_theta, X, y)

    print("Cost at initial theta (zeros): {:0.3f}".format(cost))
    print("Expected cost (approx): 0.693\n")
    print("Gradient at initial theta (zeros):")
    for g in grad:
        print("{:0.4f}".format(g))
    print("Expected gradients (approx):\n-0.1000\n-12.0092\n-11.2628\n")

    # Compute and display cost and gradient with non-zero theta
    test_theta = array([-24, 0.2, 0.2])
    [cost, grad] = costFunction(test_theta, X, y)

    print("Cost at test theta: {:0.3f}".format(cost))
    print("Expected cost (approx): 0.218")
    print("Gradient at test theta:")
    for g in grad:
        print("{:0.3f}".format(g))
    print("Expected gradients (approx):\n0.043\n2.566\n2.647\n")

    input("Program paused. Press enter to continue.")

    # ============= Part 3: Optimizing using fminunc  =============
    #  In this exercise, you will use a built-in function (fminunc) to find the
    #  optimal parameters theta.

    #  Set options for fminunc
    options = {"maxiter": 400}

    #  Run fminunc to obtain the optimal theta
    #  This function will return theta and the cost
    result = minimize(fun=costFunction,
                      x0=initial_theta,
                      args=(X, y),
                      jac=True,
                      method="TNC",
                      options=options)

    theta = result.x
    cost = result.fun

    # Print theta to screen
    print("Cost at theta found by fminunc: {:0.3f}".format(cost))
    print("Expected cost (approx): 0.203")
    print("theta:")
    for t in theta:
        print("{:0.3f}".format(t))
    print("Expected theta (approx):")
    print("-25.161\n0.206\n0.201\n")

    # Plot Boundary
    plotDecisionBoundary(theta, X, y)

    # Put some labels
    # Labels and Legend
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")

    # Specified in plot order
    plt.plot([], [], "bo", label="Admitted")
    plt.plot([], [], "r*", label="Not admitted")
    plt.legend()
    plt.show()

    input("Program paused. Press enter to continue.")

    # ============== Part 4: Predict and Accuracies ==============
    #  After learning the parameters, you"ll like to use it to predict the outcomes
    #  on unseen data. In this part, you will use the logistic regression model
    #  to predict the probability that a student with score 45 on exam 1 and
    #  score 85 on exam 2 will be admitted.
    #
    #  Furthermore, you will compute the training and test set accuracies of
    #  our model.
    #
    #  Your task is to complete the code in predict.py

    #  Predict probability for a student with score 45 on exam 1
    #  and score 85 on exam 2

    prob = sigmoid(array([1, 45, 85]) @ theta)
    print(
        "For a student with scores 45 and 85, we predict an admission probability of {:0.3f}".format(prob))
    print("Expected value: 0.775 +/- 0.002")

    # Compute accuracy on our training set
    p = predict(theta, X)

    print("Train Accuracy: {:0.1f}".format(mean(p == y) * 100))
    print("Expected accuracy (approx): 89.0")


#############################################################################

if __name__ == "__main__":
    ex2()
