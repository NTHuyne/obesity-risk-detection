from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def tune_clf_hyperparameters(clf, param_grid, X_train, y_train, scoring='accuracy', n_splits=5, refit=True):
    '''
    This function optimizes hyperparameters for a classifier by searching over a specified hyperparameter grid. 
    It utilizes GridSearchCV and cross-validation (StratifiedKFold) to evaluate various combinations of hyperparameters.
    Parameters:
        clf = base model
        param_grid = space of hyperparamters to tune
        X_train = the training data features
        y_train = the training data label
        scoring = scoring metric to use for evaluating the performance of the model (default = 'accuracy')
        n_splits = the number of folds to be used in cross-validation (default = 5)
        refit = refit the best estimator with the entire dataset after finding the best hyperparameters.
    Returns:
        best classification estimators,
        best selected hyperparameters,
        a data frame of tuning results.
    '''

    cv = StratifiedKFold(n_splits=n_splits, shuffle=False)

    clf_grid = GridSearchCV(clf, param_grid, cv=cv, scoring=scoring, refit=refit, n_jobs=-1)

    clf_grid.fit(X_train, y_train)

    best_hyperparameters = clf_grid.best_params_

    tuning_results = pd.DataFrame(clf_grid.cv_results_)

    return clf_grid.best_estimator_, best_hyperparameters, tuning_results