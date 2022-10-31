from helpers import *
from implementations import *
from predict import *
    
def main():
    degree = 8  #For the polynomial expansion
    lambda_ = 2e-7
    yb_train, input_data_train, _ = load_csv_data(data_path='Data/train.csv')
    _, input_data_test, ids_test = load_csv_data(data_path='Data/test.csv')
    print("Data is loaded.")
    input_data_train, input_data_test = preprocess(input_data_train, input_data_test, degree)
    print("Preprocessing is finished.")
    w, _ = ridge_regression(yb_train, input_data_train, lambda_)
    labels = predict_labels(w, input_data_test)
    create_csv_submission(ids=ids_test, y_pred=labels, name="Results/Submission_ridge_regression.csv")
    print("Result is stored in csv file.")

if __name__ == "__main__":
    main()