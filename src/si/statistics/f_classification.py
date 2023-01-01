from typing import Tuple, Union
import numpy as np
from scipy import stats

from si.data.dataset import Dataset

def f_classification(dataset: Dataset) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[float, float]]:
    """
    analisar a variância do nosso dataset.
    Função de scoring para problemas de classificação.
    Calcula o valor F da ANOVA de uma via para o dataset fornecido.
    O valor F permite analisar se a média entre dois ou mais grupos (factores) são significativamente diferentes.
    As amostras são agrupadas por classes.

    :param
    dataset: Dataset
        A labeled dataset

    :return:
    F: np.array, shape(n_features,)
        F scores
    p: np.array, shape(n_features,)
        p-values
    """
    classes = dataset.get_classes() #agrupa as samples/exemplos por classes
    groups = [dataset.X[dataset.y == c] for c in classes] #seleciona as samples de cada lista
    F, p = stats.f_oneway(*groups) #valores de F e p
    return F, p