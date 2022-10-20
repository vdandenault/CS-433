def compute_loss(y, tx, w):
    e = y - tx @ w
    N = y.shape[0]

    return (1/(2*N)) * e.T @ e

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
        """Computes the gradient at w.
            
        Args:
            y: numpy array of shape=(N, )
            tx: numpy array of shape=(N,2)
            w: numpy array of shape=(2, ). The vector of model parameters.
            
        Returns:
            An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
        """
        
        e = y - tx @ w
        N = y.shape[0]
        
        return -(1/N) * tx.T @ e

    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient_least_squares(y, tx, w)
        loss = compute_loss(y, tx, w)
        
        w = w - gamma * grad
        
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return ws[-1], losses[-1]

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
            """
            Generate a minibatch iterator for a dataset.
            Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
            Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
            Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
            Example of use :
            for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
                <DO-SOMETHING>
            """
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

        N = y.shape[0]
        yStoch, txStoch = next(batch_iter(y, tx, batch_size))
        e = yStoch - txStoch @ w
        
        return -(1/N) * txStoch.T @ e

    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        grad = compute_stoch_gradient(y, tx, w, len(y) // 10)
        loss = compute_loss(y, tx, w)
        
        w = w - gamma * grad
        
        # store w and loss
        ws.append(w)
        losses.append(loss)

        print("SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    
    return ws[-1], losses[-1]

def predict(tx, w):
    return tx.T @ w