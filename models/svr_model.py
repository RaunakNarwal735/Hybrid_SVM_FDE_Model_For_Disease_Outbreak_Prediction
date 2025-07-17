# models/svr_model.py

import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

def scale_features(train_df, test_df, feature_cols):
    """
    Standardize features using StandardScaler.

    Args:
        train_df (pd.DataFrame): Training set.
        test_df (pd.DataFrame): Test set.
        feature_cols (list): List of feature column names.

    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)
    X_test = scaler.transform(test_df[feature_cols].values)
    return X_train, X_test, scaler


def tune_and_train_svr(X_train, y_train, grid_search=True):
    """
    Train SVR with optional GridSearch hyperparameter tuning.

    Args:
        X_train (ndarray): Scaled training features.
        y_train (ndarray): Training targets.
        grid_search (bool): If True, perform GridSearchCV.

    Returns:
        (model, dict): Trained SVR model and parameters.
    """
    if not grid_search:
        model = SVR(kernel="rbf", C=10.0, epsilon=0.1)
        model.fit(X_train, y_train)
        return model, {"C": 10.0, "epsilon": 0.1, "gamma": "scale"}

    param_grid = {
        "C": [1, 10, 100],
        "epsilon": [0.01, 0.1, 0.5],
        "gamma": ["scale", 0.01, 0.1],
    }
    grid = GridSearchCV(
        SVR(kernel="rbf"),
        param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_
