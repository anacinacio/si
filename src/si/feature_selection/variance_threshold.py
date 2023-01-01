import numpy as np
from si.data.dataset import Dataset

class VarianceThreshold:
    """
    Variance Threshold feature selection.
    As características com uma variação do conjunto de formação inferior
    a este limiar serão removidas do conjunto de dados.

    :Parameters
    threshold: float
        O valor limite a utilizar para a selecção de características.
        As características com uma variação do conjunto de treino inferior a este limiar serão removidas.

    :Attributes
    variance: array-like, shape (n_features,)
            A variação de cada característica.
    """
    def __init__(self, threshold: float = 0.0):
        """
        Variance Threshold feature selection.
        As características com uma variação do conjunto de formação inferior
        a este limiar serão removidas do conjunto de dados.

        :param
        threshold: float
            O valor limite a utilizar para a selecção de características.
            As características com uma variação do conjunto de treino inferior a este limiar serão removidas.

        """
        if threshold < 0:
            raise ValueError("Threshold must be non-negative")
        #parametro
        self.threshold = threshold
        #atributo
        self.variable =None

    def fit (self, dataset: Dataset) -> 'VarianceThreshold':
        """
        Ajustar o modelo VarianceThreshold de acordo com os dados de treino fornecidos.
        estima/calcula a variância de cada feature; retorna o self (ele próprio)

        :param
        dataset: Dataset
        :return:
        self: object
        """
        self.variance = np.var(dataset.X, axis=0)
        return self #fit retorna a ele próprio

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Seleciona todas as features com variância superior ao threshold e retorna o X selecionado.
        Remove todas as características cuja variação não atinge o limiar.
        :param 
        dataset: Dataset
         
        :return:
        dataset: Dataset 
        """
        X = dataset.X

        features_mask = self.variance > self.threshold
        X = X[:, features_mask]
        features = np.array(dataset.features)[features_mask]
        return Dataset(X=X, y=dataset.y, features = list(features), label=dataset.label)


    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Corre o fit e depois o transform, ou seja,
        ajustar-se aos dados, depois transformá-los.

        :param
        dataset:Dataset

        :return:
        dataset: Dataset
        """
        self.fit(dataset)
        return self.transform(dataset)

if __name__ == '__main__':
    from si.data.dataset import Dataset

    dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                  [0, 1, 4, 3],
                                  [0, 1, 1, 3]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")
    dataset.X[:,0] = 0

    selector = VarianceThreshold(threshold=0.1)
    selector = selector.fit(dataset)
    dataset = selector.transform(dataset)
    print(dataset.features)


