from typing import Callable

import numpy as np

from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification

class SelectKBest:
    """
    Selecionar features(características) de acordo com os scores k mais altos
    A classificação das features é realizada através do cálculo
    dos scores de cada feature utilizando uma função scoring:
    - f_classification: ANOVA F-value between label/feature for classification tasks.
    - f_regression: F-value obtained from F-value of r's pearson correlation coefficients for regression tasks.

    :parameters
    score_func : callable
        Função de recolha do dataset e devolução de um par de arrays(scores, p_values)
    k: int, default =10
        Números de top features a selecionar
    :Attributes
    F: np.array, shape(n_features,)
        F scores of features
    p: np.array, shape(n_features,)
        p-values of F-scores
    """

    def __init__(self, score_func: Callable = f_classification, k: int = 10):
        """
        Selecionar features de acordo com os scores k mais altos
        :param
         score_func: callable
            Função de recolha do dataset e devolução de um par de arrays(scores, p_values)


         k: int, default =10
            Números de top features a selecionar
        """
        self.k = k
        self.score_func = score_func
        self.F = None
        self.p = None

    def fit (self,dataset: Dataset) -> 'SelectKBest':
        """
        estima o F e p para cada feature usando a scoring_func

        :param
        dataset: Dataset
            A labeled dataset

        :return:
        self: object
            Returns self
        """
        self.F, self.p = self.score_func(dataset) #score_fun é a nossa score_fun

    def transform(self, dataset: Dataset) -> Dataset:
        '''
        seleciona as k features com valor de F mais alto

        :param
        dataset: Dataset
            A labeled dataset

        :return:
        dataset: Dataset
            A labeled dataset with the k highest scoring features.
        '''
        idxs = np.argsort(self.F)[-self.k:]
        features = np.array(dataset.features)[idxs]
        return Dataset(X = dataset.X[:,idxs], y=dataset.y, features=list(features), label=dataset.label)

    def fit_transform(self, dataset:Dataset) -> Dataset:
        """
        corre o fit e depois o transform

        :param
        dataset: Dataset
            A labeled dataset

        :return:
        dataset: Dataset
            A labeled dataset with the k highest scoring features.
        """
        self.fit(dataset)
        return self.transform(dataset)