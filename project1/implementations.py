import numpy as np
from predict import *
from preprocessing import *
from crossvalidation import *

def compute_loss(y, tx, w):
    N = y.shape[0]

    return (1/(2*N)) * sum((y[i] - tx[i].T @ w) ** 2  for i in range(N))

def sigmoid(z):
    return 1.0/(1 + np.exp(-z))

def gradients(X, y, y_hat):    
    dw = (1/X.shape[0])*np.dot(X.T, (y_hat - y))
    db = (1/X.shape[0])*np.sum((y_hat - y)) 
    return dw, db

def normalize(X):
    return [(X - X.mean(axis=0))/X.std(axis=0) for _ in range(X.shape[1])]


def compute_loss(y, tx, w): #loss for least squares
    N = y.shape[0]
    return (1/(2*N)) * sum((y[i] - tx[i].T @ w) ** 2  for i in range(N))

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.
        
    Args:
        y: numpy array of shape=(N, ) #N is the number of rows of the train set
        tx: numpy array of shape=(N,x) #x is the number of features in the train set
        initial_w: numpy array of shape=(x, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        
    Returns:
        loss: the loss value (scalar) of the last iteration of GD
        w: the model parameters as a numpy array of shape (2, ), for the last iteration of GD 
    """

    def compute_gradient_least_squares(y, tx, w):
        N = y.shape[0]
        A = np.reshape(tx.T @ y, (tx.T.shape[0], 1))
        #reshape to make sure the following expression works as intended
        #we had issues when that reshape was not done

        return (1/N) * (tx.T @ (tx @ w) - A)
        #formula of the gradient : (1/N) * tx.T @ (tx @ w - y) 
        #computing it this way requires way too much memory for the intermediate steps (466 GB, for computing tx.t @ tx)
        #that's why the formula has been split in the implementation

    w = initial_w
    for n_iter in range(max_iters): #iterations of gradient descent
        grad = compute_gradient_least_squares(y, tx, w) #compute the gradient for the descent
        w = w - gamma * grad

        print("GD least squares iter. {bi}/{ti}".format(
              bi=n_iter+1, ti=max_iters))

    loss = compute_loss(y, tx, w) #compute the loss to show it
    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent (SGD) algorithm.
        
    Args:
        y: numpy array of shape=(N, ) #N is the number of rows of the train set
        tx: numpy array of shape=(N,x) #x is the number of features in the train set
        initial_w: numpy array of shape=(x, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        
    Returns:
        w: the model parameters as a numpy array of shape (2, ), for the last iteration of SGD 
        loss: the loss value (scalar) of the last iteration of SGD
    """
    
    def compute_stoch_gradient(y, tx, w, batch_size):
        #Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.
        
        def batch_iter(y, tx, batch_size): #compute a random mini-batch from the train set
            shuffle_indices = list(np.random.randint(0, len(y)-1) for _ in range(batch_size))
            #there might be duplicates, but given the size of the datasets we are using
            #and the relative size of the batch, compared to the size of the train set
            #duplicates are pretty rare

            shuffled_y = y[shuffle_indices]
            shuffled_tx = tx[shuffle_indices]
                
            return shuffled_y, shuffled_tx

        #prepare a random mini-batch of train data, the gradient will be computed only for this subset
        yStoch, txStoch = batch_iter(y, tx, batch_size)
        N = yStoch.shape[0]
        A = np.reshape(txStoch.T @ yStoch, (txStoch.T.shape[0], 1)) 
        #reshape to make sure the following works as intended
        #we had issues when that reshape was not done

        return (1/N) * (txStoch.T @ (txStoch @ w) - A) #formula for the gradient of the quadratic error

    w = initial_w
    for n_iter in range(max_iters): #compute the iterations of the gradient descent
        grad = compute_stoch_gradient(y, tx, w, len(y) // 100)
        w = w - gamma * grad

        print("SGD least squares iter. {bi}/{ti}".format(
              bi=n_iter+1, ti=max_iters))
    
    loss = compute_loss(y, tx, w)
    return w, loss

def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    
    A = tx.T @ tx 
    b = tx.T @ y 
    
    w = np.linalg.solve(A,b)
    
    loss = compute_loss(y,tx,w)
    
    return w, loss


def ridge_regression(y, tx, lambda_):
    """implement ridge regression.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])
    """
    N, D = tx.shape
    
    lambda_accent = 2*N*lambda_
    
    A = tx.T @ tx + lambda_accent * np.identity(D)
    b = tx.T @ y
    w = np.linalg.solve(A, b)
    
    loss = compute_loss(y, tx, w)  
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma): 
    #Logistic regression using gradient descent or SGD (y ∈ {0, 1})
    def train(X, y, epochs, lr, initial_w):
        m, _ = X.shape
        w = initial_w
        b = 0
        batch_size = 100
        y = y.reshape(m,1)

        losses = []

        for epoch in range(epochs):
            for i in range((m-1)//batch_size + 1):
                start_i = i*batch_size
                end_i = start_i + batch_size
                xb = X[start_i:end_i]
                yb = y[start_i:end_i]

                y_hat = sigmoid(np.dot(xb, w) + b)
                dw, db = gradients(xb, yb, y_hat)


                w -= lr*dw
                b -= lr*db

            print("Epoch number: " + str(epoch))
            losses.append(compute_loss(y, X, w))
        
        return w, losses[-1]
    
    return train(X=tx, y=y, epochs=max_iters, lr=gamma, initial_w=initial_w) 

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma): 
    def train(X, y, epochs, lr, initial_w):
        m, _ = X.shape
        w = initial_w
        b = 0
        batch_size = 100
        y = y.reshape(m,1)
        losses = []
        for epoch in range(epochs):
            for i in range((m-1)//batch_size + 1):
                start_i = i*batch_size
                end_i = start_i + batch_size
                xb = X[start_i:end_i]
                yb = y[start_i:end_i]
                y_hat = sigmoid(np.dot(xb, w) + b)
                dw, db = gradients(xb, yb, y_hat)
                w -= lr*(lambda_ * (dw*dw))
                b -= lr* (lambda_ * (db*db))
            print("Epoch number: " + str(epoch))
            losses.append(compute_loss(y, X, w))

        return w, losses[-1] 
    return train(X=tx, y=y, epochs=max_iters, lr=gamma, initial_w=initial_w) 

