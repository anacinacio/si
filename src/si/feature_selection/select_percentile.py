from typing import Callable

import numpy as np

from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification

class SelectPercentile:
    """

    :parameters
    score_func : callable
        Função de análise da variância
    percentile:
        Percentil para as features a selecionar
    :Attributes
    F: np.array, shape(n_features,)
        F scores of features
    p: np.array, shape(n_features,)
        p-values of F-scores
    """

    def __init__(self, score_func: Callable = f_classification, percentile: int = 10):
        """
        Selecionar features de acordo com os scores mais altos até ao percentil indicado
        :param
         score_func: callable
            Função de recolha do dataset e devolução de um par de arrays(scores, p_values)

         percentile:
            Percentil para as features a selecionar

        """
        self.percentile = percentile
        self.score_func = score_func
        self.F = None
        self.p = None

    def fit (self,dataset: Dataset) -> 'SelectPercentile':
        """
        estima o F e p para cada feature usando a scoring_func

        :param
        dataset: Dataset
            dataset

        :return:
        self: object
        """
        self.F, self.p = self.score_func(dataset) #score_fun é a nossa score_fun
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        '''
        seleciona as features com valor de F mais alto até ao percentil indicado

        :param
        dataset: Dataset
           dataset

        :return:
        dataset: Dataset
            dataset com as features (valor de F mais alto até ao percentil)
        '''
        num_features = len(dataset.features)
        percentile_features = int(num_features)*self.percentile
        idxs = np.argsort(self.F)[- percentile_features:]
        best_features = dataset.X[:,idxs]
        best_features_names = [dataset.features[i] for i in idxs]

        return Dataset(best_features, dataset.y, best_features_names, dataset.label)

    def fit_transform(self, dataset:Dataset) -> Dataset:
        """
        corre o fit e depois o transform

        :param
        dataset: Dataset

        :return:
        dataset: Dataset
        """
        self.fit(dataset)
        return self.transform(dataset)