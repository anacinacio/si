import numpy as np
from si.data.dataset import Dataset

class PCA:
    """
    Técnica de álgebra linear para reduzir as dimensões do dataset. O
    PCA a implementar usa a técnica de álgebra linear SVD (Singular
    Value Decomposition)

    :parameters
    n_components: int
        Números de componentes

    :Estimated parameters
    mean: np.array
        média das amostras
    components: np.array
        os componentes principais aka matriz unitária dos eigenvectors
    explained_variance:
        a variância explicada aka matriz diagonal dos eigenvalues

    """

    def __init__(self, n_components: int):
        """

        :param n_components: int
            Números de componentes
        :param mean: np.array
            média das amostras
        :param components:
            os componentes principais aka matriz unitária dos eigenvectors
        :param explained_variance:
            a variância explicada aka matriz diagonal dos eigenvalues
        """
        #parameters
        self.n_components = n_components

        self.mean = None
        self.components = None
        self.explained_variance = None


    def get_centering_data(self, dataset: Dataset):
        """
        Começa por centrar os dados.
        - Infere a média das amostras
        - Subtrai a média ao dataset (X – mean)

        :param dataset: Dataset
            Dataset object
        """
        self.mean = np.mean(dataset.X, axis=0)
        centering_data =  dataset.X - self.mean
        return centering_data

    def get_svd(self, dataset: Dataset):
        """
        - A função numpy.linalg.svd(X, full_matrices=False) dá-nos o U, S, VT
        :param dataset: Dataset
            Dataset object

        """
        U,S,VT = np.linalg.svd(dataset, full_matrices=False)
        return  U, S, VT

    def fit(self, dataset: Dataset):
        """
        estima a média, os componentes e a variância explicada.

        :param dataset: Dataset
            Dataset object
        :return:

        """
        #centrar os dados
        centering_data = self.get_centering_data(dataset)

        #calcula svd
        U,S,VT = self.get_svd(centering_data)

        #Infere os componentes principais
        #(Os componentes principais (components) correspondem aos primeiros n_components de VT)
        self.components = VT[:self.n_components]

        #Infere a variância explicada
        #(A variância explicada pode ser calculada pela seguinte formula EV = S2 / (n - 1) – n corresponde ao número
        #de amostras e S é dado pelo SVD)
        #(A variância explicada(explained_variance) corresponde aos primeiros n_componentes de EV)
        EV = S**2 / (len(dataset.X) - 1)
        self.explained_variance = EV[:self.n_components]

        return self

    def transform(self, dataset: Dataset):
        """
        Calcula o dataset reduzido usando os componentes principais.

        Começa por centrar os dados:
            - Subtrai a média ao dataset (X – mean)
            - Usa a média inferida no método fit
            -> get_centering_data
        Calcula o X reduzido:
            - A redução de X pode ser calculado pela seguinte formula Xreduced = X*V
            - A função numpy.dot(X, V) – multiplicação de matrizes - dá-nos a redução de X às componentes principais
            - NOTA: V corresponde à matriz transporta de VT

        :param dataset: Dataset
            Dataset object
        :return:
            X reduzido
        """
        #centrar dados
        centering_data = self.get_centering_data(dataset)

        #X reduzido
        V =  self.components.T

        Xreduced = np.dot(centering_data, V)

        return Xreduced

    def fit_transform(self, dataset: Dataset) -> np.ndarray:
        """
        Corre o fit e depois o transform, ou seja,
        ajustar-se aos dados, depois transformá-los.

        :param dataset: Dataset
            Dataset object
        :return: np.darray
            dataset transformado
        """
        self.fit(dataset)
        return self.transform(dataset)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from si.io.csv import read_csv
    iris = read_csv("C:/Users/Carolina/Documents/GitHub/si/datasets/iris.csv", sep=',',features=True, label=True)
    print(iris.X[:5])
    n = 2
    iris_pca = PCA(n)
    iris_x_red = iris_pca.fit_transform(iris)
    iris_x_red
