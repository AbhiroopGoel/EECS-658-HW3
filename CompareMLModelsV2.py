"""
EECS 658 - Assignment 3
CompareMLModelsV2.py

Description:
    This program compares 12 machine learning models on the Iris dataset.
    It uses 2-fold cross-validation so that the combined test predictions
    cover all 150 Iris samples exactly once.

    Models included:
      1) Naive Bayes (GaussianNB)
      2) Linear Regression (as classifier via rounding)
      3) Polynomial Regression degree 2 (as classifier via rounding)
      4) Polynomial Regression degree 3 (as classifier via rounding)
      5) kNN (KNeighborsClassifier)
      6) LDA (LinearDiscriminantAnalysis)
      7) QDA (QuadraticDiscriminantAnalysis)
      8) SVM (svm.LinearSVC)
      9) Decision Tree (DecisionTreeClassifier)
     10) Random Forest (RandomForestClassifier)
     11) ExtraTrees (ExtraTreesClassifier)
     12) Neural Network (MLPClassifier)

Inputs:
    None (loads Iris dataset from scikit-learn)

Outputs:
    For each of the 12 models:
      - Accuracy score
      - Confusion matrix (must sum to 150)

Collaborators:
    None

Other sources:
    Assignment 3 instructions
    Lecture slides (Regression Classifiers, Naive Bayes, LDA/QDA/kNN)
    ChatGPT

Author:
    Abhiroop Goel

Creation date:
    2026-02-24
"""

# -----------------------------
# Imports
# -----------------------------

# Import warnings so I can optionally silence convergence warnings in Codespaces.
import warnings

# Import numpy for numeric work and arrays.
import numpy as np

# Import the Iris dataset loader.
from sklearn.datasets import load_iris

# Import StratifiedKFold to do reproducible 2-fold cross-validation.
from sklearn.model_selection import StratifiedKFold

# Import PolynomialFeatures to build degree 2 and 3 feature expansions.
from sklearn.preprocessing import PolynomialFeatures

# Import LinearRegression for regression-based "classifiers".
from sklearn.linear_model import LinearRegression

# Import Gaussian Naive Bayes classifier.
from sklearn.naive_bayes import GaussianNB

# Import kNN classifier.
from sklearn.neighbors import KNeighborsClassifier

# Import MinMaxScaler to scale features into [0,1] for distance-based kNN.
from sklearn.preprocessing import MinMaxScaler

# Import Pipeline so scaling is applied correctly inside CV (fit on train only).
from sklearn.pipeline import Pipeline

# Import LDA and QDA classifiers.
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Import SVM classifier (LinearSVC).
from sklearn.svm import LinearSVC

# Import Decision Tree classifier.
from sklearn.tree import DecisionTreeClassifier

# Import Random Forest and ExtraTrees classifiers.
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

# Import neural network classifier.
from sklearn.neural_network import MLPClassifier

# Import accuracy and confusion matrix metrics.
from sklearn.metrics import accuracy_score, confusion_matrix


# -----------------------------
# Helper functions
# -----------------------------

def print_results(model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print accuracy + confusion matrix and verify the matrix totals 150."""
    # Compute accuracy score.
    acc = accuracy_score(y_true, y_pred)

    # Compute confusion matrix.
    cm = confusion_matrix(y_true, y_pred)

    # Print model name label.
    print(model_name)

    # Print accuracy rounded to 3 decimals.
    print("Accuracy Score:", round(acc, 3))

    # Print confusion matrix.
    print("Confusion Matrix:")
    print(cm)

    # Print blank line for readability.
    print("")

    # Verify the confusion matrix counts add to 150.
    total = int(cm.sum())
    if total != 150:
        print("WARNING: Confusion matrix sum is", total, "but should be 150.\n")


def clamp_regression_predictions(pred: np.ndarray) -> np.ndarray:
    """
    Convert regression predictions into class labels:
      - Round to nearest integer
      - Clamp into valid class range [0, 2]
    """
    # Round predictions to nearest integer.
    pred = pred.round()

    # Clamp values above 2 down to 2.
    pred = np.where(pred >= 3.0, 2.0, pred)

    # Clamp values below 0 up to 0.
    pred = np.where(pred <= -1.0, 0.0, pred)

    # Return integer labels.
    return pred.astype(int)


def run_regression_model(model_name: str, poly_degree: int,
                         X: np.ndarray, y: np.ndarray,
                         skf: StratifiedKFold) -> None:
    """
    Run LinearRegression as a classifier using rounding and clamping.
    This function supports degree 1, 2, or 3 using PolynomialFeatures.

    Key point:
      PolynomialFeatures must be fit on the TRAIN fold only each time.
    """
    # Create an array to store predictions for all 150 samples.
    y_pred_all = np.empty_like(y)

    # Loop over the 2 folds (train/test split indices).
    for train_idx, test_idx in skf.split(X, y):
        # Slice out training features.
        X_train = X[train_idx]

        # Slice out testing features.
        X_test = X[test_idx]

        # Slice out training labels.
        y_train = y[train_idx]

        # Create polynomial feature transformer for this fold.
        poly = PolynomialFeatures(degree=poly_degree)

        # Fit on train only, then transform train.
        X_train_poly = poly.fit_transform(X_train)

        # Transform test using the same mapping learned from train.
        X_test_poly = poly.transform(X_test)

        # Create linear regression model.
        reg = LinearRegression()

        # Fit regression on training fold.
        reg.fit(X_train_poly, y_train)

        # Predict on test fold (continuous values).
        preds = reg.predict(X_test_poly)

        # Convert continuous outputs into class labels.
        y_pred_all[test_idx] = clamp_regression_predictions(preds)

    # Print final combined results (covers all 150 samples).
    print_results(model_name, y, y_pred_all)


def run_classifier_model(model_name: str, clf,
                         X: np.ndarray, y: np.ndarray,
                         skf: StratifiedKFold) -> None:
    """
    Run a true classifier under 2-fold CV.
    Predictions from both test folds cover all 150 samples.
    """
    # Create an array to store predictions for all 150 samples.
    y_pred_all = np.empty_like(y)

    # Loop over 2 folds.
    for train_idx, test_idx in skf.split(X, y):
        # Slice out training data.
        X_train = X[train_idx]

        # Slice out testing data.
        X_test = X[test_idx]

        # Slice out training labels.
        y_train = y[train_idx]

        # Fit the model on the training fold.
        clf.fit(X_train, y_train)

        # Predict on the testing fold.
        y_pred_all[test_idx] = clf.predict(X_test)

    # Print final combined results (covers all 150 samples).
    print_results(model_name, y, y_pred_all)


# -----------------------------
# Main program
# -----------------------------

def main() -> None:
    # Optional: hide convergence warnings so your console output looks clean.
    warnings.filterwarnings("ignore")

    # Load iris dataset.
    iris = load_iris()

    # Extract feature matrix.
    X = iris.data

    # Extract class labels (0,1,2).
    y = iris.target

    # Build a reproducible 2-fold stratified splitter.
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=1)

    # 1) Naive Bayes (GaussianNB)
    run_classifier_model(
        "Naive Bayesian (GaussianNB)",
        GaussianNB(),
        X, y, skf
    )

    # 2) Linear Regression (degree 1)
    run_regression_model(
        "Linear Regression (LinearRegression as classifier)",
        poly_degree=1,
        X=X, y=y, skf=skf
    )

    # 3) Polynomial Regression degree 2
    run_regression_model(
        "Polynomial Regression Degree 2 (LinearRegression as classifier)",
        poly_degree=2,
        X=X, y=y, skf=skf
    )

    # 4) Polynomial Regression degree 3
    run_regression_model(
        "Polynomial Regression Degree 3 (LinearRegression as classifier)",
        poly_degree=3,
        X=X, y=y, skf=skf
    )

    # 5) kNN (use k = 9, and scale features to [0,1] inside a pipeline)
    knn_pipeline = Pipeline([
        ("scaler", MinMaxScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=9))
    ])
    run_classifier_model(
        "kNN (KNeighborsClassifier, k=9, Euclidean distance + MinMax scaling)",
        knn_pipeline,
        X, y, skf
    )

    # 6) LDA
    run_classifier_model(
        "LDA (LinearDiscriminantAnalysis)",
        LinearDiscriminantAnalysis(),
        X, y, skf
    )

    # 7) QDA
    run_classifier_model(
        "QDA (QuadraticDiscriminantAnalysis)",
        QuadraticDiscriminantAnalysis(),
        X, y, skf
    )

    # 8) SVM (LinearSVC)
    run_classifier_model(
        "SVM (LinearSVC)",
        LinearSVC(random_state=1),
        X, y, skf
    )

    # 9) Decision Tree
    run_classifier_model(
        "Decision Tree (DecisionTreeClassifier)",
        DecisionTreeClassifier(random_state=1),
        X, y, skf
    )

    # 10) Random Forest
    run_classifier_model(
        "Random Forest (RandomForestClassifier)",
        RandomForestClassifier(random_state=1),
        X, y, skf
    )

    # 11) ExtraTrees
    run_classifier_model(
        "ExtraTrees (ExtraTreesClassifier)",
        ExtraTreesClassifier(random_state=1),
        X, y, skf
    )

    # 12) Neural Network (MLPClassifier)
    run_classifier_model(
        "Neural Network (MLPClassifier)",
        MLPClassifier(random_state=1, max_iter=2000),
        X, y, skf
    )


if __name__ == "__main__":
    main()
