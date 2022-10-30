from helpers import *
from implementations import *
from preprocessing import *
from predict import *
from crossvalidation import *
import numpy as np

import pickle, os

def main():
    print("Reading the data")
    if not os.path.exists("data.p"):
        yb_train, input_data_train, ids_train = load_csv_data(data_path='Data/train.csv')
        yb_test, input_data_test, ids_test = load_csv_data(data_path='Data/test.csv')
        print("Data is loaded")

        pickle.dump((yb_train, input_data_train, ids_train, yb_test, input_data_test, ids_test), open("data.p", "wb"))
    else:
        yb_train, input_data_train, ids_train, yb_test, input_data_test, ids_test = pickle.load(open("data.p", "rb"))
        print("Sunglasses on")

    seed = 15

    initial_w = np.zeros((input_data_train.shape[1],1))
    #set of parameters A : max_iters, gamma, lambda_ = 500, 1e-10, 0.5
    #set of parameters B : max_iters, gamma, lambda_ = 5000, 1e-10, 0.5
    max_iters = 50000
    gamma = 3e-7
    #gamma = 3e-7 pour GD et SGD
    lambda_ = 0.5

    print("time to train!")
    w, loss = least_squares_GD(yb_train, input_data_train, initial_w, max_iters, gamma)
    #w, loss = least_squares_SGD(yb_train, input_data_train, initial_w, max_iters, gamma)
    print(w, loss)
    #w, _ = logistic_regression(yb_train, input_data_train, initial_w, max_iters, gamma)
    #print(_)
    #labels = predict_logistic_regression(input_data_test.T, w)

    #w, _ = reg_logistic_regression(yb_train, input_data_train, lambda_, initial_w, max_iters, gamma)
    #labels = predict_logistic_regression(input_data_test.T, w)
    labels = predict_least_squares(w, input_data_test.T)

    create_csv_submission(ids=ids_test, y_pred=labels, name="Results/Submission_least_squares_GD_50000.csv")

if __name__ == "__main__":
    main()