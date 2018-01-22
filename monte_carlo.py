# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 16:11:09 2018

@author: junbai

Generate series for Monte-Carlo Simulation
"""
import sys
sys.path.append("C:/Users/j291414/my algorithms")

import datapy as dy
reload(dy)

# include seed
import numpy as np
import pandas as pd


def generate_correlated_2D(n, mu1, sigma1, mu2, sigma2, rho):
    n1 = np.random.normal(size=n)
    n2 = np.random.normal(size=n)
    
    Y1 = mu1 + sigma1 * n1
    Y2 = mu2 + sigma2 * (rho * n1 + np.sqrt(1 - rho**2) * n2)
    
    return Y1, Y2


def generate_random_walk_2D(n, base1, mu1, sigma1, base2, mu2, sigma2, rho, lognormal=False):
    Y1, Y2 = generate_correlated_2D(n, mu1, sigma1, mu2, sigma2, rho)
    
    if not lognormal:
        Y1_hat = np.cumsum(Y1)
        Y2_hat = np.cumsum(Y2)
        return base1+Y1_hat, base2+Y2_hat
    else:
        Y1_hat = np.cumprod(np.exp(Y1))
        Y2_hat = np.cumprod(np.exp(Y2))
        return base1*Y1_hat, base2*Y2_hat
    
    
def check_corr(trials):
    result = list()
    
    for i in range(trials):
        Y1, Y2 = generate_random_walk_2D(252, 100, 0, 10, 100, 0, 15, 0.95)
        result.append(dy.cross_correlation(Y1, Y2, model=('diff', 'diff'), suppress=False))
        
    return np.array(result)