import numpy as np
import pandas as pd

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection


if __name__ == "__main__":
    # read data
    df = pd.read_csv('mobile_train.csv')

    # here we have training features
    X = df.drop("price_range", axis=1).values
    # target
    y = df.price_range.values

    # define the model
    classifier = ensemble.RandomForestClassifier(n_jobs=-1)

    # define the grid search    
    param_grid = {
        "n_estimators": np.arange(100, 1500, 100),
        "max_depth": np.arange(1, 31),
        "criterion": ["gini", "entropy"]
    }

    # initialize the grid search
    model = model_selection.RandomizedSearchCV(
        estimator=classifier,
        param_distributions=param_grid,
        n_iter=20,
        scoring="accuracy",
        verbose=10,
        n_jobs=1,
        cv=5
    )

    # fit the model ans extract best score
    model.fit(X, y)
    print(f"Best score: {model.best_score_}")

    print(f"Best parameters set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print(f"\t{param_name}: {best_parameters[param_name]}")