import numpy as np

''' O mse é uma medida de erro, mede o valor de erros estimados e valores reais.
Usa a formula: sum((y_pred - y_true )**2) / (m*2) 
- m representa o número de amostras
- h(x(i)) representa os valores estimados
- Y(i) representa os valores reais
Soma as diferenças entre os valores estimados e os valores reais ao quadrado e divide pelo número de amostras vezes 2
Serve para saber se o nosso modelo chegou perto ou não do que pretendiamos.

Uma medida para no final de treinar o modelo, saber se chegou perto daquilo que é a realidade ou não'''

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    It returns the mean squared error of the model on the given dataset

    :parameter
    y_true: np.ndarray
        valores reais de Y
    y_pred: np.ndarray
        valores estimados de Y

    :return
    mse: float
        O valor do erro entre y_true e y_pred
    """
    return np.sum((y_true - y_pred) ** 2) / (len(y_true) * 2)

def mse_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Retorna a derivada do erro quadrático médio para a variável y_pred.

    :parameter
    y_true: np.ndarray
        valores reais de Y
    y_pred: np.ndarray
        valores estimados de Y

    :return
    mse_derivative: np.ndarray
        A derivada do erro médio ao quadrado
    """
    return -2 * (y_true - y_pred) / (len(y_true) * 2)
