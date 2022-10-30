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

    #input_data_train, input_data_test = preprocess(input_data_train, input_data_test, 2)
    _, n = input_data_train.shape
    initial_w = np.zeros((n,1))
    max_iters = 1000
    gamma = 0.001
    #w, losses = logistic_regression(yb_train, input_data_train, initial_w, max_iters, gamma)
    #labels = predict_logistic_regression(w, input_data_test)
    #plot_loss_function(losses, max_iters)

    degree = 8
    lambda_ = 0.001
    nb_k = 5
    k_indices = build_k_indices(yb_train, nb_k, 42)

    train_accs = np.zeros((nb_k,))
    test_accs = np.zeros((nb_k,))

    print("Time to train!")
    for k in range(nb_k):
        train_accs[k], test_accs[k] = cross_validation(yb_train, input_data_train,  k_indices, k, degree, logistic_regression, initial_w, max_iters, gamma)        
        print("Fold:", k , " Training accuracy:",  train_accs[k], " Test accuracy: ", test_accs[k])

    print("Average test accuracy: ",  np.average(test_accs))


    #create_csv_submission(ids=ids_test, y_pred=labels, name="Results/Submission_logistic_regression_1000.csv")

if __name__ == "__main__":
    main()