import numpy as np
from implementations import reg_logistic_regression

def hyperparameter_search_rlr(yb_train, input_data_train, initial_w, max_iters, lamdbas, gammas): 
    losses = []
    weigths = []
    for gamma in gammas: 
        print("GAMMA: " + str(gamma))
        for lambda_ in lamdbas:
            print("lambda_: " + str(lambda_))
            w, loss = reg_logistic_regression(yb_train, input_data_train, lambda_, initial_w, max_iters, gamma)
            losses.append(loss)
            weigths.append(w)
            print("CURRENT LOSSS: " + str(loss))
    idx = np.argmin(losses)
    return weigths[idx], losses[idx]