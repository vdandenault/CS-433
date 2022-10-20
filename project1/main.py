from helpers import load_csv_data, create_csv_submission
#from logistic_regression import logistic_regression
from implementations import *
import numpy as np

import pickle, os

def main():
    print("start")
    if not os.path.exists("data.p"):
        yb_train, input_data_train, ids_train = load_csv_data(data_path='../Data/train.csv')
        yb_test, input_data_test, ids_test = load_csv_data(data_path='../Data/test.csv')
        print("data is loaded")

        pickle.dump((yb_train, input_data_train, ids_train, yb_test, input_data_test, ids_test), open("data.p", "wb"))
    else:
        yb_train, input_data_train, ids_train, yb_test, input_data_test, ids_test = pickle.load(open("data.p", "rb"))
        print("sunglasses on")

    initial_w = np.zeros((input_data_train.shape[1],1))
    inital_b = np.zeros((input_data_train.shape[1],1))
    max_iters = 1000
    gamma = 0.01
    lambda_ = 0.5

    print("time to train!")
    #w,b, losses = logistic_regression(yb_train, input_data_train, inital_b, initial_w, max_iters, gamma)
    #w, loss = least_squares_GD(yb_train, input_data_train, initial_w, max_iters, gamma)
    #w, loss = least_squares(yb_train, input_data_train)
    w, loss = ridge_regression(yb_train, input_data_train, lambda_)

    print(w)
    print(loss)
    #print(b)
    #print(losses)

if __name__ == "__main__":
    main()