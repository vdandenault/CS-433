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
        yb_train, input_data_train, ids_train = load_csv_data(data_path='../Data/train.csv')
        yb_test, input_data_test, ids_test = load_csv_data(data_path='../Data/test.csv')
        print("Data is loaded")

        pickle.dump((yb_train, input_data_train, ids_train, yb_test, input_data_test, ids_test), open("data.p", "wb"))
    else:
        yb_train, input_data_train, ids_train, yb_test, input_data_test, ids_test = pickle.load(open("data.p", "rb"))
        print("Sunglasses on")

    seed = 15

    initial_w = np.zeros((input_data_train.shape[1],1))
    max_iters = 100
    gamma = 1e-10
    degree = 8
    lambda_ = 0.001


    nb_k = 5
    k_indices = build_k_indices(yb_train, nb_k, seed)

    train_accs = np.zeros((nb_k,))
    test_accs = np.zeros((nb_k,))


    print("Time to train!")
    for k in range(nb_k):
        train_accs[k], test_accs[k] = cross_validation(yb_train, input_data_train,  k_indices, k, degree, ridge_regression, lambda_)
        print("Fold:", k , " Training accuracy:",  train_accs[k], " Test accuracy: ", test_accs[k])
    
    print("Average test accuracy: ",  np.average(test_accs))
        

    #w, _ = logistic_regression(yb_train, input_data_train, initial_w, max_iters, gamma)
    #labels = predict_logistic_regression(w, input_data_test.T)

    #w, _ = reg_logistic_regression(yb_train, input_data_train, lambda_, initial_w, max_iters, gamma)
    #labels = predict_least_squares(w, input_data_test.T)

    x_train, x_test = preprocess(input_data_train, input_data_test, degree)
    
    weights, _ = ridge_regression(yb_train, x_train, lambda_)

    labels = predict_labels(weights, x_test)

    create_csv_submission(ids=ids_test, y_pred=labels, name="Results/Submission_ridge_regression")


if __name__ == "__main__":
    main()