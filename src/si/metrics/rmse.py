import math

import numpy as np

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula a o erro seguindo a formula da RMSE (RMQE em português):
    RMSE = √(∑((ytrue-ypred)^2)/N)
    N - representa o número de amostras
    :parameter
    y_true: np.ndarray
        valores reais de Y
    y_pred: np.ndarray
        valores estimados de Y

    :return
    rmse: float
        O valor do erro entre y_true e y_pred
    """
    return math.sqrt(np.sum(((y_true - y_pred)**2) / len(y_true)))

