import numpy as np

def poly_features(X, power=8):
    """
    POLYFEATURES Maps X (1D vector) into the p-th power
      [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
      maps each example into its polynomial features where
      X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];

    """
    X_poly = np.zeros((X.shape[0],power))

    """% ====================== YOUR CODE HERE ======================
    Instructions: Given a vector X, return a matrix X_poly where the p-th 
            column of X contains the values of X to the p-th power.
    """

    X_poly = [X ** p for p in range(power + 1)]

    return np.vstack(X_poly).T
