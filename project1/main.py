from helpers import *
from implementations import *
from preprocessing import *
from predict import *
from crossvalidation import *
import numpy as np

import pickle, os

def hyperparameter_search(yb_train, input_data_train, initial_w, max_iters, lamdbas, gammas): 
    losses = []
    weigths = []
    for gamma in gammas: 
        for lambda_ in lamdbas:
            w, loss = reg_logistic_regression(yb_train, input_data_train, lambda_, initial_w, max_iters, gamma)
            losses.append(loss)
            weigths.append(w)
            print("CURRENT LOSSS: " + str(loss))
    idx = np.argmin(losses)
    return weigths[idx], losses[idx]

    
def main():
    print("Reading the data")
    if not os.path.exists("data.p"):
        yb_train, input_data_train, ids_train = load_csv_data(data_path='../Data/train.csv')
        yb_test, input_data_test, ids_test = load_csv_data(data_path='../Data/test.csv')
        print("Data is loaded")

        pickle.dump((yb_train, input_data_train, ids_train, yb_test, input_data_test, ids_test), open("data.p", "wb"))
    else:
        yb_train, input_data_train, ids_train, yb_test, input_data_test, ids_test = pickle.load(open("data.p", "rb"))

    initial_w = np.zeros((input_data_train.shape[1],1))
    #set of parameters A : max_iters, gamma, lambda_ = 500, 1e-10, 0.5
    #set of parameters B : max_iters, gamma, lambda_ = 5000, 1e-10, 0.5
    max_iters = 100
    gamma = 3e-7 #gamma = 3e-7 pour GD et SGD
    lambda_ = 0.5

    print("time to train!")
    #w, loss = least_squares_GD(yb_train, input_data_train, initial_w, max_iters, gamma)
    #w, loss = least_squares_SGD(yb_train, input_data_train, initial_w, max_iters, gamma)
    #print(w, loss)
    #w, loss = logistic_regression(yb_train, input_data_train, initial_w, max_iters, gamma)
    #print(loss)
    #labels = predict_logistic_regression(input_data_test.T, w)
    x_train, x_test = preprocess(input_data_train, input_data_test, degree=1)
    lambas = np.arange(0.1, 0.9, 0.05).tolist()
    gammas = np.arange(1e-7, 1e-6, 1e-7).tolist()

    #w, loss = hyperparameter_search(yb_train, input_data_train, initial_w, max_iters, lambas, gammas)
    w, loss = reg_logistic_regression(yb_train, x_train, lambda_, initial_w, max_iters, gamma)
    labels = predict_logistic_regression(w, x_test)
    print(w.shape)
    print("MINIMUM LOSS: " + str(loss))

    create_csv_submission(ids=ids_test, y_pred=labels, name="Results/Submission_least_squares_SGD_10000.csv")

if __name__ == "__main__":
    main()