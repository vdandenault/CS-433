import numpy as np

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x


def build_poly(x, degree):
    """
    Polynomial basis functions for input data x, for j=0 up to j=degree.

    Args:
        x: numpy array of shape (N, dim); N is the number of samples.
        degree: integer.
        
    Returns:
        poly: numpy array of shape (N,dim*d+1)

    """
    
    phi = np.hstack((np.ones((len(x),1)), x))
    if degree > 1: 
        for i in range(2, degree+1):
            phi = np.c_[phi, (x ** i)]
    return phi


def preprocess(input_data_train, input_data_test, degree):
    """ A function to preprocess the data """

    # Replace missing values(val == -999) with the median of that column(=feature).

    for i, feature in enumerate(input_data_train.T):
            median = np.median(feature[feature != -999])
            input_data_train[:,i] = np.where(feature == -999, median, feature)
            input_data_test[:,i] = np.where(input_data_test[:,i] == -999, median, input_data_test[:,i])
        

    # Remove the outliers in the features by replacing them with a less extreme value.
    perc = 7
    for i, feature in enumerate(input_data_train.T):
        input_data_train[:, i] = np.where(feature < np.percentile(feature,perc), np.percentile(feature,perc), feature)
        input_data_train[:, i] = np.where(feature > np.percentile(feature,100-perc), np.percentile(feature,100-perc), feature)

    for i, feature in enumerate(input_data_test.T):
        input_data_test[:, i] = np.where(feature < np.percentile(feature,perc), np.percentile(feature,perc), feature)
        input_data_test[:, i] = np.where(feature > np.percentile(feature,100-perc), np.percentile(feature,100-perc), feature)

    # Build a polynomial basis 
    x_train = build_poly(input_data_train, degree)
    x_test = build_poly(input_data_test, degree)

    return x_train, x_test
