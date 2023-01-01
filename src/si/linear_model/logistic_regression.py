import numpy as np
import matplotlib.pyplot as plt

from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.sigmoid_function import sigmoid_function
from si.statistics.sigmoid_function import sigmoid_function
from si.metrics.accuracy import accuracy

class LogisticRegression:
    """

    :parameter
    l2_penalty: float
        o coeficiente da regularização L2
    alpha: float
        a learning rate (taxa de aprendizagem)
    max_iter: int
        número máximo de iterações

    :attributes
    theta: np.array
        Os coeficientes/parâmetros do modelo para as variáveis de entrada (features)
        Por exemplo, x0 * theta[0] + x1 * theta[1] + ...
    theta_zero: float
        o coeficiente/parâmetro zero. Também conhecido como interceção
        Por exemplo, theta_zero * 1
    cost_history: dict
    """
    def __init__(self, l2_penalty: float = 1, alpha: float = 0.001, max_iter: int = 2000):
        """
        :parameter
        l2_penalty: float
            o coeficiente da regularização L2
        alpha: float
            a learning rate (taxa de aprendizagem)
        max_iter: int
            número máximo de iterações
        """
        #parametros
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter

        #atributos
        self.theta = None
        self.theta_zero = None
        self.cost_history = None

    def fit_new(self, dataset: Dataset) -> 'LogisticRegression':
        m, n = dataset.shape()

        #inicializar os parâmetros do modelo
        self.theta = np.zeros(n)
        self.theta_zero = 0

        #cost history -> dicionário
        self.cost_history = {}

        #Gradient descent
        for i in range(self.max_iter):
            #valor previsto y
            y_predit = np.dot(dataset.X, self.theta) + self.theta_zero

            #aplicar sigmoid function
            y_pred = sigmoid_function(y_predit)

            #computação e actualização do gradiente com a taxa de aprendizagem
            gradient = (self.alpha * (1 / m)) * np.dot(y_pred - dataset.y, dataset.X)

            #cálculo do penalty
            penalization_term = self.alpha * (self.l2_penalty / m) * self.theta

            #actualização dos parâmetros do modelo
            self.theta = self.theta - gradient - penalization_term
            self.theta_zero = self.theta_zero - (self.alpha * (1 / m)) * np.sum(y_pred - dataset.y)

            #cost
            self.cost_history[i]=self.cost(dataset)
            if i > 0 and (self.cost_history[i-1] - self.cost_history[i]) < 0.0001:
                self.alpha = self.alpha/2



    def fit_old(self, dataset: Dataset) -> 'LogisticRegression':
        """
        Adaptar o modelo ao dataset

        :param
        dataset: Dataset
            O dataset para se adaptar ao modelo

        :return
        self: LogisticRegression
            O modelo adaptado
        """
        m, n = dataset.shape()

        #inicializar os parâmetros do modelo
        self.theta = np.zeros(n)
        self.theta_zero = 0

        #cost history -> dicionário
        self.cost_history = {}

        #Gradient descent
        for i in range(self.max_iter):
            #valor previsto y
            y_predit = np.dot(dataset.X, self.theta) + self.theta_zero

            #aplicar sigmoid function
            y_pred = sigmoid_function(y_predit)

            #computação e actualização do gradiente com a taxa de aprendizagem
            gradient = (self.alpha * (1 / m)) * np.dot(y_pred - dataset.y, dataset.X)

            #cálculo do penalty
            penalization_term = self.alpha * (self.l2_penalty / m) * self.theta

            #actualização dos parâmetros do modelo
            self.theta = self.theta - gradient - penalization_term
            self.theta_zero = self.theta_zero - (self.alpha * (1 / m)) * np.sum(y_pred - dataset.y)

            #cost
            self.cost_history[i]=self.cost(dataset)
            if i > 0 and (self.cost_history[i-1] - self.cost_history[i]) < 0.0001:
                break



    def fit(self, dataset: Dataset) -> 'RidgeRegression':

        if self.fit_new:
            print('new')
            self.fit_new(dataset)
        else:
            self.fit_old(dataset)

        return self

    def predict(self, dataset: Dataset) -> np.array:
        """
        estima a variável de saída (dependente) usando os thetas estimados

        :param
        dataset: Dataset
            The dataset para prever

        :return
        predictions: np.array
            A previsão do dataset
        """
        predictions = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero)

        mask = predictions >= 0.5 #meio de sigmoid
        predictions[mask] = 1
        predictions[~mask] = 0
        return predictions

    def score(self, dataset: Dataset) -> float:
        """
        calcula o erro entre as previsões e os valores reais

        :param
        dataset: Dataset
            Dataset para calcular a accuracy

        :return
        accuracy: float
            accuracy entre os valores reais e as previsões
        """
        y_pred = self.predict(dataset)
        return accuracy(dataset.y, y_pred)

    def cost(self, dataset: Dataset) -> float:
        """
        calcula a função de custo entre as previsões e os valores reais

        :param
        dataset: Dataset
            Dataset para calcular cost function

        :return
        cost: float
            A função de custo do modelo
        """
        y_pred = self.predict(dataset)
        predictions = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero)
        cost = (-dataset.y * np.log(predictions)) - ((1 - dataset.y)* np.log(1 - predictions))
        cost = np.sum(cost) / dataset.shape()[0]
        cost = cost + (self.l2_penalty * np.sum(self.theta**2)/( 2 * dataset.shape()[0]))
        return cost

    def plot(self):
        """
        permite visualizar o comportamento do custo em função do número de iterações.
        """

        plt.plot(self.cost_history.keys(), self.cost_history.values())
        plt.xlabel("Iterações")
        plt.ylabel("Custo")
        plt.show()


if __name__ == '__main__':
    # import dataset
    from si.data.dataset import Dataset
    from si.model_selection.split import train_test_split

    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)
    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

    # fit the model
    model = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=1000)
    model.fit(dataset_train)


    # compute the score
    score = model.score(dataset_test)
    print(f"Score: {score}")
