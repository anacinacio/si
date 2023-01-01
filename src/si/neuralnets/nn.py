import numpy as np

from typing import Callable
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.metrics.mse import mse, mse_derivative

class NN:
    """
    O NN é o modelo de Rede Neural.
    Compreende a topologia modelo incluindo várias camadas de redes neuronais.
    O algoritmo de adaptação do modelo é baseado na retropropagação.
    :parameter
        layers: list
            Lista de camadas na rede neural
        epochs: int
            Número de epochs para treinar o modelo
        learning_rate: float
            A taxa de aprendizagem do modelo
        loss_function: Callable
            A função loss a utilizar
        loss_derivative: Callable
            A derivada da função loss a utilizar
        verbose: bool

    :attributes
        history: dict
            A history do modelo treinado.
    """
    def __unit__(self, layers:list, epochs: int=1000, loss_function: Callable = mse,
                 learning_rate: float =0.01, loss_derivation: Callable = mse_derivative, verbose: bool = False):
        """
        Inicializar o modelo neural network

        :param
        layers: list
            Lista de camadas na rede neural
        epochs: int
            Número de epochs para treinar o modelo
        learning_rate: float
            A taxa de aprendizagem do modelo
        loss_function: Callable
            A função loss a utilizar
        loss_derivative: Callable
            A derivada da função loss a utilizar
        verbose: bool
        """
        #parametros
        self.layers = layers
        self.epochs = epochs
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.loss_derivation = loss_derivation
        self.verbose = verbose

        #atributos
        self.history ={}

    def fit(self, dataset: Dataset) -> "NN":
        """
        onde treinamos o modelo com o dataset
        iteramos cada camada
        :param
        dataset: Dataset
            O dataset
        :return:
        self: NN

        """
        X = dataset.X
        y = dataset.Y #predict_y

        for epoch in range(1, self.epochs + 1):

            #propagação para a frente
            #x = dataset.X.copy()
            for layer in self.layers:
                X = layer.forward(X)

            #propagação para trás
            error = self.loss_derivation(y,X)
            for layer in self.layers[::-1]:
                error = layer.backward(error, self.learning_rate)

            #salver histórico
            cost = self.loss_function(y, X)
            self.history[epoch] = cost

            #loss
            if self.verbose:
                print(f'Epoch {epoch}/{self.epochs} - {cost}')

        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        :param
        dataset: Dataset
            o dataset

        :return
        predictions: np.ndarray
        """
        X = dataset.X

        #propagação para a frente
        for layer in self.layers:
            X = layer.forward(X)

        return X

    def cost(self, dataset: Dataset) -> float:
        """
        calcula o custo do modelo no dataset

        :param
        dataset: Dataset
            dataset

        :return
        cost: float
            custo do modelo
        """
        y_pred = self.predict(dataset)
        return self.loss_function(dataset.y, y_pred)

    def score(self, dataset: Dataset, scoring_func: Callable = accuracy) -> float:
        """
        Calcula o score do modelo no dataset

        :param
        dataset: Dataset
            o dataset
        scoring_func: Callable
            função de scoring

        :return
        score: float
            O score do modelo
        """
        y_pred = self.predict(dataset)
        return scoring_func(dataset.y, y_pred)




if __name__ == '__main__':
    from neuronetworks.layer import Dense
    from neuronetworks.sigmoid_activation import SigmoidActivation
    from neuronetworks.soft_max_activation import SoftMaxActivation
    from neuronetworks.re_lu_activation import ReLUActivation

    X = np.array([[0,0],
                [0,1],
                [1,0],
                [1,2]])

    Y = np.array([1,
                0,
                0,
                1])

    dataset = Dataset(X, Y, ['x1', 'x2'], 'x1 XNOR x2')
    print(dataset.to_dataframe())


    w1 = np.array([[20, -20],
                    [20, -20]])

    b1 = np.array([[-30, 10]])


    l1 = Dense(input_size = 2, output_size=2)
    l1.weights = w1
    l1.bias = b1


    w2 = np.array([[20],
                    [20]])

    b2 = np.array([[-10]])

    l2 = Dense(input_size = 2, output_size=1)
    l2.weights = w2
    l2.bias = b2

    l1_sa = SigmoidActivation()
    l2_sa = SigmoidActivation()

    nn_model_sa = NN(layers=[l1, l1_sa, l2, l2_sa])
    nn_model_sa.fit(dataset=dataset)


    print(nn_model_sa.predict(dataset))

