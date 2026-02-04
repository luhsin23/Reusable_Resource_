import numpy as np

def parameters_linear(d):
    n = 2 * d
    Lambda = np.tile(np.array([3, 2]), d)
    mu_inv = np.ones(n)
    q = np.append(4, np.ones(d-1) * 5)
    r = np.tile(np.array([3, 1]), d)
    LP = 10 * d
    return n, Lambda, mu_inv, q, r, LP

def parameters_2H(d):
    n = 2 * d
    Lambda = np.tile(np.array([3, 2]), d)
    mu_inv = np.ones(n)
    q = np.append(np.ones(d-1) * 4, d+3)
    r = np.tile(np.array([3, 1]), d)
    LP = 10 * d
    return n, Lambda, mu_inv, q, r, LP

def parameters_non_hier():
    n = 4
    Lambda = np.array([3,2,2,2])
    mu_inv = np.ones(n)
    q = np.array([5,2,2])
    r = np.array([3,1,1,1])
    LP = 18
    return n, Lambda, mu_inv, q, r, LP