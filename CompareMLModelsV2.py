"""
EECS 658 - Assignment 3
CompareMLModelsV2.py

Compares 12 ML models on the Iris dataset using 2-fold cross-validation.

Models:
  1) Naive Baysian (GaussianNB)
  2) Linear regression (LinearRegression)
  3) Polynomial of degree 2 regression (LinearRegression)
  4) Polynomial of degree 3 regression (LinearRegression)
  5) kNN (KNeighborsClassifier)
  6) LDA (LinearDiscriminantAnalysis)
  7) QDA (QuadraticDiscriminantAnalysis)
  8) SVM (svm.LinearSVC)
  9) Decision Tree (DecisionTreeClassifier)
 10) Random Forest (RandomForestClassifier)
 11) ExtraTrees (ExtraTreesClassifier)
 12) NN (neural_network.MLPClassifier)

For each model, prints:
  - Confusion matrix (must sum to 150)
  - Accuracy metric

Author: Abhiroop Goel
KU ID: 3067979
Creation date: 2026-02-26
Other sources: Lecture slides, ChatGPT
"""

import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import confusion_matrix, accuracy_score


def print_results(model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(model_name)
    print("Accuracy Score:", round(acc, 3))
    print("Confusion Matrix:")
    print(cm)
    print("")

    if int(cm.sum()) != 150:
        print("WARNING: Confusion matrix sum is", int(cm.sum()), "but should be 150.\n")


def clamp_regression_predictions(pred: np.ndarray) -> np.ndarray:
    pred = pred.round()
    pred = np.where(pred >= 3.0, 2.0, pred)
    pred = np.where(pred <= -1.0, 0.0, pred)
    return pred.astype(int)


def run_regression_classifier(model_name: str, poly_degree: int,
                              X1: np.ndarray, y1: np.ndarray,
                              X2: np.ndarray, y2: np.ndarray) -> None:
    """
    2-fold approach:
      Train on fold1 -> test fold2
      Train on fold2 -> test fold1
    Concatenate results to make a 150-sample test result.
    """
    # Fold1 -> Fold2
    poly1 = PolynomialFeatures(degree=poly_degree)
    X1_poly = poly1.fit_transform(X1)
    X2_poly = poly1.transform(X2)

    reg1 = LinearRegression()
    reg1.fit(X1_poly, y1)
    pred_fold2 = clamp_regression_predictions(reg1.predict(X2_poly))

    # Fold2 -> Fold1 (IMPORTANT: fit PolynomialFeatures on fold2 for this direction)
    poly2 = PolynomialFeatures(degree=poly_degree)
    X2_poly_b = poly2.fit_transform(X2)
    X1_poly_b = poly2.transform(X1)

    reg2 = LinearRegression()
    reg2.fit(X2_poly_b, y2)
    pred_fold1 = clamp_regression_predictions(reg2.predict(X1_poly_b))

    actual = np.concatenate([y2, y1])
    predicted = np.concatenate([pred_fold2, pred_fold1])

    print_results(model_name, actual, predicted)


def run_classifier(model_name: str, clf,
                   X1: np.ndarray, y1: np.ndarray,
                   X2: np.ndarray, y2: np.ndarray) -> None:
    """
    2-fold approach:
      Train on fold1 -> test fold2
      Train on fold2 -> test fold1
    Concatenate results to make a 150-sample test result.
    """
    clf.fit(X1, y1)
    pred_fold2 = clf.predict(X2)

    clf.fit(X2, y2)
    pred_fold1 = clf.predict(X1)

    actual = np.concatenate([y2, y1])
    predicted = np.concatenate([pred_fold2, pred_fold1])

    print_results(model_name, actual, predicted)


def main() -> None:
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Match the common lecture-style split used in earlier work:
    # fixed random_state makes the experiment reproducible.
    X_fold1, X_fold2, y_fold1, y_fold2 = train_test_split(
        X, y, test_size=0.50, random_state=1
    )

    # 1) Naive Bayes (default params)
    run_classifier("Naive Baysian (GaussianNB)", GaussianNB(),
                   X_fold1, y_fold1, X_fold2, y_fold2)

    # 2) Linear Regression (degree 1)
    run_regression_classifier("Linear regression (LinearRegression)", 1,
                              X_fold1, y_fold1, X_fold2, y_fold2)

    # 3) Polynomial degree 2 regression
    run_regression_classifier("Polynomial degree 2 regression (LinearRegression)", 2,
                              X_fold1, y_fold1, X_fold2, y_fold2)

    # 4) Polynomial degree 3 regression
    run_regression_classifier("Polynomial degree 3 regression (LinearRegression)", 3,
                              X_fold1, y_fold1, X_fold2, y_fold2)

    # 5) kNN (default params, default k=5, no scaling)
    run_classifier("kNN (KNeighborsClassifier)", KNeighborsClassifier(),
                   X_fold1, y_fold1, X_fold2, y_fold2)

    # 6) LDA (default params)
    run_classifier("LDA (LinearDiscriminantAnalysis)", LinearDiscriminantAnalysis(),
                   X_fold1, y_fold1, X_fold2, y_fold2)

    # 7) QDA (default params)
    run_classifier("QDA (QuadraticDiscriminantAnalysis)", QuadraticDiscriminantAnalysis(),
                   X_fold1, y_fold1, X_fold2, y_fold2)

    # 8) SVM (default params)
    run_classifier("SVM (svm.LinearSVC)", LinearSVC(),
                   X_fold1, y_fold1, X_fold2, y_fold2)

    # 9) Decision Tree (default params)
    run_classifier("Decision Tree (DecisionTreeClassifier)", DecisionTreeClassifier(),
                   X_fold1, y_fold1, X_fold2, y_fold2)

    # 10) Random Forest (default params)
    run_classifier("Random Forest (RandomForestClassifier)", RandomForestClassifier(),
                   X_fold1, y_fold1, X_fold2, y_fold2)

    # 11) ExtraTrees (default params)
    run_classifier("ExtraTrees (ExtraTreesClassifier)", ExtraTreesClassifier(),
                   X_fold1, y_fold1, X_fold2, y_fold2)

    # 12) Neural Network (default params)
    run_classifier("NN (neural_network.MLPClassifier)", MLPClassifier(),
                   X_fold1, y_fold1, X_fold2, y_fold2)


if __name__ == "__main__":
    main()
