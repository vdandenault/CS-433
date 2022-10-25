from helpers import load_csv_data, create_csv_submission
from implementations import predict_least_squares, ridge_regression, least_squares_GD, least_squares,  \
        predict_logistic_regression, logistic_regression, reg_logistic_regression
import numpy as np

import pickle, os

def main():
    print("Reading the data")
    if not os.path.exists("data.p"):
        yb_train, input_data_train, ids_train = load_csv_data(data_path='Data/train.csv')
        yb_test, input_data_test, ids_test = load_csv_data(data_path='Data/test.csv')
        print("data is loaded")

        pickle.dump((yb_train, input_data_train, ids_train, yb_test, input_data_test, ids_test), open("data.p", "wb"))
    else:
        yb_train, input_data_train, ids_train, yb_test, input_data_test, ids_test = pickle.load(open("data.p", "rb"))
        print("sunglasses on")

    initial_w = np.zeros((input_data_train.shape[1],1))
    max_iters = 100
    gamma = 1e-10
    lambda_ = 0.5

    print("time to train!")
    w, _ = least_squares_GD(yb_train, input_data_train, initial_w, max_iters, gamma)
    #w, _ = logistic_regression(yb_train, input_data_train, initial_w, max_iters, gamma)
    #labels = predict_logistic_regression(input_data_test.T, w)

    #w, _ = reg_logistic_regression(yb_train, input_data_train, lambda_, initial_w, max_iters, gamma)
    labels = predict_least_squares(input_data_test.T, w)

    create_csv_submission(ids=ids_test, y_pred=labels, name="Results/Submission_least_squares_GD")

if __name__ == "__main__":
    main()