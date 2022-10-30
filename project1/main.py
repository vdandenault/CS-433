from helpers import *
from implementations import *
from preprocessing import *
from predict import *
from crossvalidation import *
import numpy as np
from hyperparameter_search import *
from plotting import plot_loss_function
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


    _, n = input_data_train.shape
    initial_w = np.zeros((n,1))
    max_iters = 100
    w, losses = logistic_regression(yb_train, input_data_train, initial_w, max_iters, 0.01)
    labels = predict_logistic_regression(w, input_data_test)
    plot_loss_function(losses, max_iters)
    #create_csv_submission(ids=ids_test, y_pred=labels, name="Results/Submission_least_squares_SGD_10000.csv")

if __name__ == "__main__":
    main()