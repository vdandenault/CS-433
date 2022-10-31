from helpers import *
from implementations import *
from preprocessing import *
from predict import *
from crossvalidation import *
import numpy as np
from hyperparameter_search import *
from plotting import plot_loss_function
    
def main():
    yb_train, input_data_train, _ = load_csv_data(data_path='Data/train.csv')
    _, input_data_test, ids_test = load_csv_data(data_path='Data/test.csv')
    
    #Delete correlated features
    #input_data_train[:, [26, 27, 28]] = 0

    #Pre-processing
    #input_data_train, input_data_test = preprocess(input_data_train, input_data_test, 2)

    _, n = input_data_train.shape
    initial_w = np.zeros((n,1))
    max_iters = 100
    w, losses = logistic_regression(yb_train, input_data_train, initial_w, max_iters, 0.001)
    labels = predict_logistic_regression(w, input_data_test)
    plot_loss_function(losses, max_iters)
    create_csv_submission(ids=ids_test, y_pred=labels, name="Results/Submission_logistic_regression_100.csv")

if __name__ == "__main__":
    main()