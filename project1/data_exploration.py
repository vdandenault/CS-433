from helpers import *
from plotting import plot_bar_from_counter


def main():
    yb_train, input_data_train, ids_text = load_csv_data(data_path='Data/train.csv')
    yb_test, input_data_test, ids_test = load_csv_data(data_path='Data/test.csv')
    plot_bar_from_counter(yb_train, title="Target Distribution in Training Data")
    plot_bar_from_counter(yb_test, title="Target Distribution in Test Data")

if __name__ == "__main__":
    main()