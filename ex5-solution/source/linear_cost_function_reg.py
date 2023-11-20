import numpy as np

def linear_cost_function_reg(theta, X, y, lambda_):
    """
    LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
    regression with multiple variables
      [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
      cost of using theta as the parameter for linear regression to fit the
      data points in X and y. Returns the cost in J and the gradient in grad

    """
    m = X.shape[0]
    cost=0
    gradient = np.zeros(theta.shape)
    """====================== YOUR CODE HERE ======================
    Instructions: Compute the cost and gradient of regularized linear 
                  regression for a particular choice of theta.
    
                  You should set J to the cost and grad to the gradient.
    
    """

    h = X.dot(theta)
    errors = h - y
    cost = sum(errors ** 2) / (2.0 * m)

    reg_cost = (lambda_ / (2.0 * m)) * np.sum(theta[1:] ** 2)
    cost = cost + reg_cost

    gradient = (1.0 / m) * X.T.dot(errors)
    reg_gradient = (lambda_ / m) * theta
    reg_gradient[0] = 0

    gradient = gradient + reg_gradient

    return cost, gradient