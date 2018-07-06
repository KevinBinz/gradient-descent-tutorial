# Gradient Descent Tutorial

## Introduction

**Optimization** is the art of finding parameters that comprise the best predictive model. To do this, you must define a loss function as a way to quantify model performance. In linear regression, this loss function is **mean squared error (MSE)** also known as Euclidean distance. 

An optimizer takes training data and discovers these 

- **Random Search** (lib/random_search.py). For n iterations, choose parameters at random, and perform an argmax: keep the best ones found so far.
- **Grid Search** (TBA). For n iterations, explore parameter space in a grid pattern, and perform an argmax: keep the parameters that yield the best score. 
- **Gradient Descent** (lib/gradient_descent_2d). For n iterations, use local slope information (the **gradient**) to decide how to adjust the parameter settings. In parameter space, the result looks something like "rolling down a hill". 

These iterative methods can be contrasted with the following **analytic** approach. 

- **Ordinary Least Squares (OLS)** (lib/least_squares.py). In one iteration, use linear algebra "black magic" to jump to the best parameters. I explain the mathematics of this approach [here](https://kevinbinz.com/2017/07/02/ols-via-projection/).
  - Unfortunately, OLS is *not* available in most situations, e.g., in logistic regression after you inject a non-linear link function. That said, it is available in linear regression, and provides a useful point of comparison to iterative methods like gradient descent. 

## How To Run

To view the results of all of these algorithms (and compare/contrast) just go to main.py and hit Run! :)
