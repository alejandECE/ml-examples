#  Created by Luis Alejandro (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import numpy as np


# Function to compute the sigmoid function
def sigmoid(x):
  return 1. / (1. + np.exp(-x))


# Function to compute the model output
def compute_output(X, w):
  return sigmoid(X.dot(w))


# Function to compute cost in Logistic Regression
def compute_cost(y, h):
  entropy = -sum([np.log(prob) if label else np.log(1. - prob) for prob, label in zip(h, y)])
  return entropy


# Function to compute gradient in Logistic Regression
def compute_gradient(X, y, h):
  residual = h - y
  return X.T.dot(residual)


# Function to compute Hessian in Logistic Regression
def compute_hessian(X, h):
  S = np.diag(h * (1. - h))
  H = X.T.dot(S).dot(X)
  return H
