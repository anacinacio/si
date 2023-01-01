import numpy as np

def sigmoid_function(X: np.ndarray) -> np.ndarray:
    """
    Retorna a função sigmóide do input dado

    :param X: np.ndarray
        valores de entrada
    :return:
    sigmoid: np.ndarray
        A probabilidade dos valores serem iguais a 1 (função sigmoid)
    """
    return (1/(1 + np.exp(-X)))
