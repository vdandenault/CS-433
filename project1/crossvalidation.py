import numpy as np
from predict import accuracy, predict_labels, predict_logistic_regression
from implementations import *
from helpers import *
from preprocessing import * 


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing. If ratio times the number of samples is not round
    you can use np.floor. Also check the documentation for np.random.permutation,
    it could be useful.
    
    Args:
        x: numpy array of shape (N,), N is the number of samples.
        y: numpy array of shape (N,).
        ratio: scalar in [0,1]
        seed: integer.
        
    Returns:
        x_tr: numpy array containing the train data.
        x_te: numpy array containing the test data.
        y_tr: numpy array containing the train labels.
        y_te: numpy array containing the test labels.
    """

    np.random.seed(seed)
    
    rand_indixes = np.random.permutation(len(y))
    split = int(np.floor(ratio * len(y)))
    train, test = rand_indixes[:split], rand_indixes[split:] 

    x_tr, x_te = x[train], x[test]
    y_tr, y_te = y[train], y[test]

    return x_tr, x_te, y_tr, y_te


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.
    
    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, k_indices, k, degree, method, *args):

    """
    Return the test and train accuracy of a given method for a fold corresponding to k_indices
    
    Args:
        y:          shape=(N,)
        x:          shape=(N, dim), dim = number of features
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()
        degree:     scalar, cf. build_poly()
        method

    """
    test = k_indices[k]
    train = np.ravel(k_indices[k])
    train = np.array([train[i] for i in range(len(train)) if i != k])

    x_train = x[train, :]
    x_test = x[test, :]
    y_train = y[train]
    y_test = y[test]

    x_train, x_test = preprocess(x_train, x_test, degree)

    if args == []:
        w, _ = method(y_train, x_train)
    else:
        if len(args) == 3:
            #the first one is initial_w, we have to redefine it since there are more features in the preprocessed data
            args = list(args)
            args[0] = np.zeros((x_train.shape[1], 1))

        w, _ = method(y_train, x_train, *args)
       
    train_labels = predict_labels(w, x_train)
    test_labels = predict_labels(w, x_test)
        
    train_acc = accuracy(train_labels, y_train)
    test_acc = accuracy(test_labels, y_test)
    
    return train_acc, test_acc

if __name__ == "__main__": #run crossvalidation on algorithms
    import os, pickle

    if not os.path.exists("data.p"):
        yb_train, input_data_train, ids_train = load_csv_data(data_path='../Data/train.csv')
        yb_test, input_data_test, ids_test = load_csv_data(data_path='../Data/test.csv')
        print("Data is loaded")
        pickle.dump((yb_train, input_data_train, ids_train, yb_test, input_data_test, ids_test), open("data.p", "wb"))
    else:
        yb_train, input_data_train, ids_train, yb_test, input_data_test, ids_test = pickle.load(open("data.p", "rb"))

    seed = 15

    def nbFeaturesPoly(x):
        nb = x


    initial_w = np.zeros((input_data_train.shape[1],1)) #for least_squares
    max_iters = 5000
    gamma = 3e-7

    degree = 8
    lambda_ = 0.001

    nb_k = 5
    k_indices = build_k_indices(yb_train, nb_k, seed)

    train_accs = np.zeros((nb_k,))
    test_accs = np.zeros((nb_k,))

    print("Time to train!")
    for k in range(nb_k):
        #ridge regression
        #train_accs[k], test_accs[k] = cross_validation(yb_train, input_data_train,  k_indices, k, degree, ridge_regression, lambda_)

        #least_squares GD
        train_accs[k], test_accs[k] = cross_validation(yb_train, input_data_train,  k_indices, k, degree, least_squares_GD, initial_w, max_iters, gamma)
        
        #least_squares SGD
        #train_accs[k], test_accs[k] = cross_validation(yb_train, input_data_train,  k_indices, k, degree, least_squares_SGD, initial_w, max_iters, gamma)
        
        print("Fold:", k , " Training accuracy:",  train_accs[k], " Test accuracy: ", test_accs[k])

    print("Average test accuracy: ",  np.average(test_accs))