from typing import Callable, Union

import numpy as np

from si.data.dataset import Dataset
from si.statistics.euclidean_distance import euclidean_distance
from si.metrics.rmse import rmse

class KNNRegressor:
    """
    KNN Regressor -> problemas de regressão
    Estima um valor médio dos k exemplos mais semelhantes.

    :Parameters
    k: int
        número de k exemplos a considerar
    distance: Callable
        função que calcula a distância entre amostra e as amostras do dataset de treino

    :Attributes
    dataset: Dataset
        armazena o dataset de treino
    """

    def __init__(self, k: int = 1 , distance: Callable = euclidean_distance):
        """
        Inicializar KNN regressor
        :param k:int
            número de k exemplos a considerar
        :param distance:Callable
            função que calcula a distância entre amostra e as amostras do dataset de treino

        """
        self.k = k
        self.distance = distance

    def fit(self, dataset: Dataset) -> 'KNNRegressor':
        """
        armazena o dataset de treino
        :param
        dataset: Dataset

        :return:
            self: KNNRegressor
                the fitted model
        """
        self.dataset = dataset
        return self

    def _get_closest_label(self, sample: np.ndarray) -> Union[int,str]:
        """
         Devolve label mais próximo da amostra dada
        :param
        sample: np.ndarray

        :return:
            A média das labels dos k nearest neighbors
        """
        #calcular a distância entre a sample e o dataset
        distances = self.distance(sample, self.dataset.X)

        #obter os k nearest neighbors
        k_nearest_neighbors = np.argsort(distances)[:self.k]

        #obter as labels dos k nearest neighbors
        k_nearest_neighbors_labels = self.dataset.y[k_nearest_neighbors]

        return np.mean(k_nearest_neighbors_labels)

    def predict(self, dataset: Dataset)-> np.ndarray:
        """
        Prevê as classes do dataset
        :param
        dataset: Dataset

        :return:
        predictions: np.ndarray
            As previsões do modelo
        """
        return np.apply_along_axis(self._get_closest_label, axis=1, arr=dataset.X)

    def score(self, dataset: Dataset) -> float:
        """
        Returna o erro entre as previsões e os valores reais

        :param
        dataset: Dataset

        :return:
        rmse: float
            Erro calculo do erro entre as previsões e os valores reais

        """
        predictions = self.predict(dataset)
        return rmse(dataset.y, predictions)


if __name__ == '__main__':
    from si.model_selection.split import train_test_split

    # carregar e dividir o dataset(treino e teste)
    dataset_ = Dataset.from_random(600, 100, 2)
    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

    # initialize the KNN regressor
    knn = KNNRegressor(k=3)

    # estimar o dataset para treino
    knn.fit(dataset_train)

    # avaliar o modelo no dataset do teste
    score = knn.score(dataset_test)
    print(f'The accuracy of the model is: {score}')
