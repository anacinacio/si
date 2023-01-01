import numpy as np

def euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calcula a distância euclidiana entre X e Y usando a seguinte formula:
        distance_y1n = np.sqrt((x1 - y11)^2 + (x2 - y12)^2 + ... + (xn - y1n)^2)
        distance_y2n = np. sqrt((x1 - y21)^2 + (x2 - y22)^2 + ... + (xn - y2n)^2)
    :param x:
        uma amostra
    :param y:
        várias amostras

    :return:
    np.ndarray
        array com distância entre X e as várias amostras de Y
    """

    return np.sqrt(((x - y) ** 2).sum(axis=1))

