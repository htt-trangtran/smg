############################
# written by Trang H. Tran and Lam M. Nguyen
############################

"""
Different learning rates
"""

import math 

def constant(eta):
    x = lambda t : eta 
    return x

def exponential(eta, alpha):
    return lambda t : eta * (alpha ** t)

def cosine (eta, T):
    return lambda t : eta * 0.5 * (1 + math.cos(t*math.pi/T))

def diminishing(eta, alpha):
    return lambda t : eta / (alpha + t)**(1/3)


