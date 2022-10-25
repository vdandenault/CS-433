import numpy as np

def compute_loss(y, tx, w):
    N = y.shape[0]

    #return (1/(2*N)) * np.dot(e.T, e)
    return (1/(2*N)) * sum((y[i] - tx[i].T @ w) ** 2  for i in range(N))

def sigmoid(z):
    return 1.0/(1 + np.exp(-z))

def gradients(X, y, y_hat):    
    dw = (1/X.shape[0])*np.dot(X.T, (y_hat - y))
    db = (1/X.shape[0])*np.sum((y_hat - y)) 
    return dw, db

def normalize(X):
    return [(X - X.mean(axis=0))/X.std(axis=0) for _ in range(X.shape[1])]

def predict_logistic_regression(X, w):
    preds = np.array(sigmoid(predict(X, w)))
    pred_class = [1 if i > 0.5 else -1 for i in preds]
    return pred_class

def predict(tx, w):
    return tx.T @ w

def predict_least_squares(X, w):
    preds = predict(X, w)
    pred_class = [1 if i > 0.5 else -1 for i in preds]
    return pred_class

def compute_loss(y, tx, w): #loss for least squares
    N = y.shape[0]
    return (1/(2*N)) * sum((y[i] - tx[i].T @ w) ** 2  for i in range(N))

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        
    Returns:
        loss: the loss value (scalar) of the last iteration of GD
        w: the model parameters as a numpy array of shape (2, ), for the last iteration of GD 
    """

    def compute_gradient_least_squares(y, tx, w):
        N = y.shape[0]
        return -(1/N) * sum(tx[i].T * (y[i] - tx[i].T @ w) for i in range(N)) #formula

    # Define parameters to store w and loss
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient_least_squares(y, tx, w) #compute the gradient for the descent
        #loss = compute_loss(y, tx, w) #compute the loss to show it
        
        w = w - gamma * grad

        #print("GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
        #      bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    loss = compute_loss(y, tx, w) #compute the loss to show it

    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent (SGD) algorithm.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        
    Returns:
        w: the model parameters as a numpy array of shape (2, ), for the last iteration of SGD 
        loss: the loss value (scalar) of the last iteration of SGD
    """
    
    def compute_stoch_gradient(y, tx, w, batch_size):
        """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.
            
        Args:
            y: numpy array of shape=(N, )
            tx: numpy array of shape=(N,2)
            w: numpy array of shape=(2, ). The vector of model parameters.
            
        Returns:
            A numpy array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
        """
        
        def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
            data_size = len(y)

            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_y = y[shuffle_indices]
                shuffled_tx = tx[shuffle_indices]
            else:
                shuffled_y = y
                shuffled_tx = tx
                
            for batch_num in range(num_batches):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                if start_index != end_index:
                    yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

        #prepare a random mini-batch of train data, the gradient will be computed only for this subset
        yStoch, txStoch = next(batch_iter(y, tx, batch_size))
        N = yStoch.shape[0]
        return -(1/N) * sum(txStoch[i].T * (yStoch[i] - txStoch[i].T @ w) for i in range(N))

    # Define parameters to store w and loss
    w = initial_w
    
    for n_iter in range(max_iters):
        grad = compute_stoch_gradient(y, tx, w, len(y) // 10)
        #loss = compute_loss(y, tx, w)
        
        w = w - gamma * grad
        

        #print("SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
        #      bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    
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

