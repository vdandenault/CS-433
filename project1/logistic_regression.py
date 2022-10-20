import numpy as np

def sigmoid(z):
    return 1.0/(1 + np.exp(-z))

def loss(y, y_hat):
    loss = -np.mean(y*(np.log(y_hat)) - (1-y)*np.log(1-y_hat))
    return loss

def gradients(X, y, y_hat):    
    dw = (1/X.shape[0])*np.dot(X.T, (y_hat - y))
    db = (1/X.shape[0])*np.sum((y_hat - y)) 
    return dw, db

def normalize(X):
    return [(X - X.mean(axis=0))/X.std(axis=0) for _ in range(X.shape[1])]

def predict(X, w, b):
    X = normalize(X)
    preds = sigmoid(np.dot(X, w) + b)
    return np.array(preds)

def train(X, y, bs, epochs, lr, initial_w):
    m, _ = X.shape
    w = initial_w
    b = 0

    y = y.reshape(m,1)
    X = normalize(X)
    
    losses = []
    
    for _ in range(epochs):
        for i in range((m-1)//bs + 1):
            start_i = i*bs
            end_i = start_i + bs
            xb = X[start_i:end_i]
            yb = y[start_i:end_i]
            
            y_hat = sigmoid(np.dot(xb, w) + b)
            dw, db = gradients(xb, yb, y_hat)
            
            w -= lr*dw
            b -= lr*db
        
        l = loss(y, sigmoid(np.dot(X, w) + b))
        losses.append(l)
        
    return w, b, losses

def logistic_regression(y, tx, inital_bs, initial_w, max_iters, gamma): 
    #Logistic regression using gradient descent or SGD (y ∈ {0, 1})
    w, b, losses = train(X=tx, y=y, bs=inital_bs, epochs=max_iters, lr=gamma, initial_w=initial_w) 
    
    return w,b, losses