# -*- coding: utf-8 -*-
"""cross validation function."""

import numpy as np
import matplotlib.pyplot as plt
from costs import *

def ridge_regression(y, tx, lamb):
    """implement ridge regression."""
    return np.linalg.solve((tx.T @ tx) + lamb*np.eye(tx.shape[1]), tx.T @ y)

def cross_validation_visualization(lambds, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k, lambda_):
    """return the loss of ridge regression."""
    
    # get k'th subgroup in test, others in train
    train_indices = np.ravel(k_indices[np.arange(len(k_indices)) != k])
    x_train, y_train = x[train_indices], y[train_indices]
    x_test, y_test = x[k_indices[k]], y[k_indices[k]]
    
    
    # ridge regression
    w = ridge_regression(y_train, x_train, lambda_)
    
    # calculate the loss for train and test data
    loss_tr = compute_rmse(y_train, x_train, w)
    loss_te = compute_rmse(y_test, x_test, w)
    return loss_tr, loss_te

def cross_validation_demo(y, x, k_fold, lambdas, seed):
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    # cross validation: 
    for l in lambdas:
        sum_tr = 0
        sum_te = 0
        for k in range(k_fold):
            loss_tr, loss_te = cross_validation(y, x, k_indices, k, l) 
            sum_tr = sum_tr + loss_tr
            sum_te = sum_te + loss_te
        rmse_tr.append(sum_tr/k_fold)
        rmse_te.append(sum_te/k_fold)
    cross_validation_visualization(lambdas, rmse_tr, rmse_te)
    return rmse_te

