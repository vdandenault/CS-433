import numpy as np

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def accuracy(pred, val):
    """
    Computes accuracy for the predication pred of val
    """
    total_acc = 0
    for i in range(len(pred)):
        if pred[i] == val[i]:
            total_acc += 1
    acc = total_acc / len(pred)
    return acc

def predict_least_squares(w, tx):
    """
    Gives predictions given weights w and test data tx for least squares
    """
    preds = tx @ w #-> shape (N, 1) where N is the number of rows of data
    pred_class = [1 if preds[i][0] > 0 else -1 for i in range(len(preds))] 
    return pred_class

def predict_logistic_regression(w, x):
    """
    Gives predictions given weights w and test data tx for logistic regression
    """
    preds = np.array(sigmoid(np.dot(x,w)))
    pred_class = [1 if i > 0.5 else -1 for i in preds]
    return pred_class

def predict_labels(w, tx):
    """
    Gives predictions given weights w and test data tx
    """
    preds = np.dot(tx, w)
    pred_class = [1 if i > 0 else -1 for i in preds]
    return pred_class