import numpy as np

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Retorna a precisão do modelo sobre o dado dataset

    :parameter
    y_true: np.ndarray
        valores reais de Y

    y_pred: np.ndarray
        valores estimados de Y
    :return
        accuracy: float
            O valor do erro entre y_true e y_pred
    """
    return np.sum(y_true == y_pred) / len(y_true)