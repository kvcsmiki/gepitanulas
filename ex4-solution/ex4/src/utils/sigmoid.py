# -*- coding: utf-8 -*-
from numpy import exp
import numpy as np


def sigmoid(z):
    """
    computes the sigmoid of z.
    """
    g = 0
    """
    ====================== YOUR CODE HERE ======================
    Instructions: Compute the sigmoid of each value of z (z can be a matrix, vector or scalar).
    """
    
    if type(z) is list :        
        zz = np.append((z[0] * 1.0), z[1:])
        g = (1 / (1 + exp(-zz))).T
    else :
       g = 1 / (1 + exp(-(z))) 
# =============================================================
    return g