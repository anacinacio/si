from typing import Callable
import numpy as np

from si.data.dataset import Dataset
from si.statistics.euclidean_distance import euclidean_distance

class KMeans:
    """
    k-means agrupa amostras em grupos chamados centroids.
    O algoritmo tenta reduzir a distância entre as amostras e o centroid.

    :parameters
    k: int
        Números de clusters

    max_iter: int
        Número máximo de iterações

    distance: Callable
        função que calcula a distância

    :Attributes
    centroids: np.array
        média das amostras em cada centroid
    labels: np.array
        vetor com a label de cada centroid
    """

    def __init__(self, k: int, max_iter: int = 1000, distance: Callable = euclidean_distance):
        """
        K-means clustering algorithm.

        :param
        k: int
            Números de clusters

        max_iter:int
            Número máximo de iterações

        distance:Callable
            função que calcula a distância
        """
        #parameters
        self.k = k
        self.max_iter = max_iter
        self.distance = distance

        #attributes
        self.centroids = None
        self.labels = None

    def _init_centroids(self, dataset: Dataset):
        """
        Gera os primeiros k centroids.
        (usando o np.random.permutation)

        :param
        dataset: Dataset
            Dataset object
        """
        seeds = np.random.permutation(dataset.shape()[0])[:self.k]
        self.centroids = dataset.X[seeds]


    def _get_closest_centroid(self, sample: np.ndarray) -> np.ndarray:
        """
        Obter o centróide mais próximo de cada ponto de dados

        :param
        sample: np.ndarray, shape=(n_features,)
            a sample

        :return: np.ndarray
             O centroide mais próximo de cada ponto de dados
        """
        centroids_distances = self.distance(sample, self.centroids)
        closest_centroid_index = np.argmin(centroids_distances, axis = 0) #0 para dar a menor distancia
        return closest_centroid_index

    def fit(self, dataset: Dataset) -> 'KMeans':
        """
        Estima os k-means clustering
        Infere os centroids minimizando a distância entre as amostras e o centroid

        :param
        dataset: Dataset
            Dataset object
        :return: kmeans
            kmeans object
        """
        #gerar os centroides iniciais
        self._init_centroids(dataset)


        # ajustar k-means
        convergence = False
        i = 0
        labels = np.zeros(dataset.shape()[0])
        while not convergence and i < self.max_iter:
            #Obter o centroid mais próximo
            new_labels = np.apply_along_axis(self._get_closest_centroid, axis=1, arr=dataset.X)

            #calcular os novos centroides
            centroids = []
            for j in range(self.k):
                centroid = np.mean(dataset.X[new_labels == j], axis=0)
                centroids.append(centroid)


            self.centroids = np.array(centroids)

            #verificar se os centroides mudaram
            convergence = np.any(new_labels != labels)

            #substituir labels
            labels = new_labels

            #contagem por incremento
            i += 1

        self.labels = labels
        return self


    def _get_distances(self, sample: np.ndarray ) -> np.ndarray:
        """
        Calcula a distância entre cada amostra e o centróide mais próximo.

        :param sample: np.ndarray, shape = (n_features,)
            A sample
        :return: np.ndarray
             Distâncias entre cada amostra e o centróide mais próximo
        """
        return self.distance(sample, self.centroids)

    def transform(self, dataset: Dataset) -> np.ndarray:
        """
        Transforma o dataset
        Calcula a distãncia entre cada amostra e o centróide mais próximo

        :param dataset: Dataset
            Dataset object
        :return: np.darray
            dataset transformado
        """
        centroids_distances = np.apply_along_axis(self._get_distances, axis=1, arr=dataset.X)
        return centroids_distances

    def fit_transform(self, dataset: Dataset) -> np.ndarray:
        """
        Corre o fit e depois o transform, ou seja, ajustar-se aos dados, depois transformá-los.

        :param dataset: Dataset
            Dataset object
        :return: np.darray
            dataset transformado
        """
        self.fit(dataset)
        return self.transform(dataset)


    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Calcula a distância entre uma amostra e os vários centroids
        Ou seja, infere qual dos centroids está mais perto da amostra

        :param dataset: Dataset
            Dataset object

        :return:
        np.ndarray
            labels
        """
        return np.apply_along_axis(self._get_closest_centroid, axis=1, arr=dataset.X)

    def fit_predict(self, dataset: Dataset) -> np.ndarray:
        """
        Faz o fit e o predict

        :param dataset: Dataset
            Dataset object

        :return:
        np.ndarray
            labels
        """
        self.fit(dataset)
        return self.predict(dataset)

if __name__ == '__main__':
    from src.si.data.dataset import Dataset

    dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                  [0, 1, 4, 3],
                                  [0, 1, 1, 3]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")

    k_ = 2
    kmeans = KMeans(k_)
    res = kmeans.fit_transform(dataset)
    predictions = kmeans.predict(dataset)
    print(res.shape)
    print(predictions.shape)

