from typing import Callable, Union

import numpy as np

from si.data.dataset import Dataset
from si.statistics.euclidean_distance import euclidean_distance
from si.metrics.accuracy import accuracy

class KNNClassifier:
    """
    KNN Classifier
    O algoritmo k-nearest neighbors estima a classe para uma amostra tendo como base os k exemplos mais semelhantes.

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
        Inicializar KNN classifier
        :param k:int
            número de k exemplos a considerar
        :param distance:Callable
            função que calcula a distância entre amostra e as amostras do dataset de treino

        """
        self.k = k
        self.distance = distance
        self.dataset = None

    def fit(self, dataset: Dataset) -> 'KNNClassifier':
        """
        armazena o dataset de treino
        :param
        dataset: Dataset

        :return:
            self: KNNClassifier
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
        label: str or int
            A label mais próxima
        """
        #calcular a distância entre a sample e o dataset
        distances = self.distance(sample, self.dataset.X)

        #obter os k nearest neighbors
        k_nearest_neighbors = np.argsort(distances)[:self.k]

        #obter as labels dos k nearest neighbors
        k_nearest_neighbors_labels = self.dataset.y[k_nearest_neighbors]

        #obter a label mais comum
        #(classe mais comum:np.unique (retorna um array que contem, e outro array com os valores das posições))
        labels, counts = np.unique(k_nearest_neighbors_labels, return_counts=True)
        return labels[np.argmax(counts)]

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
        it returns the accuracy of the model on the given dataset

        :param
        dataset: Dataset

        :return:
        accuracy: float
            accuracy do modelo
        """
        predictions = self.predict(dataset)
        return accuracy(dataset.y, predictions)



if __name__ == '__main__':
    # importar dataset
    from si.data.dataset import Dataset
    from si.model_selection.split import train_test_split
    from si.metrics.accuracy import accuracy


    # carregar e dividir o dataset
    dataset = Dataset.from_random(600, 100, 2)

    dataset_train, dataset_test = train_test_split(dataset, test_size=0.2)

    # initialize the KNN classifier
    knn = KNNClassifier(k=3)

    # estimar o dataset para treino
    knn.fit(dataset_train)

    # avaliar o modelo no dataset do teste
    score = knn.score(dataset_test)
    print(f'The accuracy of the model is: {score}')

