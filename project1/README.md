# Machine Learning Project 1 - Higgs Boson Challenge

This repository contains Project 1 of the Machine Learning (CS-433) course given at EPFL.

## Team Members
- Jacob Schillemans
- Vincent Dandenault
- Fabrice Nemo

## Project structure: 

The project is structured in the following way:

```markdown
- _run.py_ produces the predictions of the best submission to the competition system (ridge regression). The predictions are saved in a file "Submission_ridge_regression.csv" saved in the Results folder.
- _implementations.py_ contains the implementations of the 6 Machine Learning functions asked in the assignment (least_squares_GD, least_squares_SGD, least_squares, ridge_regression, logistic_regression, reg_logistic_regression).
- _preprocessing.py_ contains functions for preprocessing the data (regularization and building polynomial features from the initial features).
- _predict.py_ contains functions for making predictions (classification) from the results given by our models.
- _helpers.py_ contains functions for reading and writing CSV files
- _crossvalidation.py_ is an implementation of cross validation as seen in lectures
- _plotting.py_ is used to plot the evolution of the loss function depending on the number of iterations of a given iterative algorithm.
- _hyperparameter\_search.py_ contains a function that does a grid search for the hyperparameters of reg_logistic_regression.
- _data\_exploration.py_ contains a function for plotting the distribution of the labels in the train and test set.

- **Data/** contains the data stored in CSV files
- **Results/** is used to store results/predictions in CSV files
```

## Instructions to run:
```markdown
.
```


