from helpers import *
from implementations import *
from predict import *
from plotting import plot_loss_function
    
def main():
    max_iters = 100
    gamma = 0.1
    yb_train, input_data_train, _ = load_csv_data(data_path='Data/train.csv')
    _, input_data_test, ids_test = load_csv_data(data_path='Data/test.csv')
    initial_w = np.zeros((input_data_train.shape[1],1))
    w, losses = logistic_regression(yb_train, input_data_train, initial_w=initial_w, max_iters=max_iters, gamma=gamma)
    labels = predict_logistic_regression(w, input_data_test)
    create_csv_submission(ids=ids_test, y_pred=labels, name="Results/Submission_logisitic_regression_1000.csv")
    plot_loss_function(losses, max_iters)

if __name__ == "__main__":
    main()