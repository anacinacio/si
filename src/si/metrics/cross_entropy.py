import numpy as np


def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    medida de erro cross entropy
    :parameter
    y_true: np.ndarray

    y_pred: np.ndarray

    Returns:
        cross entropy
    """
    return - np.sum(y_true) * np.log(y_pred) / len(y_true)


def cross_entropy_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    derivada da medida de erro cross entropy

    :parameter
    y_true: np.ndarray

    y_pred: np.ndarray

    Returns:
        derivada cross entropy
    """

    return - (y_true / y_pred) + ((1 - y_true) / (1 - y_pred)) / len(y_true)