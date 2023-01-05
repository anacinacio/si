import numpy as np
import matplotlib.pyplot as plt

from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.sigmoid_function import sigmoid_function

class LogisticRegression:
    """
    O LogisticRegression é um modelo logistico que utiliza a regularização L2.
    Este modelo resolve o problema da regressão logistica utilizando uma técnica de Gradient Descent.

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
    def __init__(self, l2_penalty: float = 1, alpha: float = 0.001, max_iter: int = 2000, adaptive: bool=False):
        """
        :parameter
        l2_penalty: float
            o coeficiente da regularização L2
        alpha: float
            a learning rate (taxa de aprendizagem)
        max_iter: int
            número máximo de iterações
        cost_history: dict
        """
        #parametros
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.adaptive = adaptive

        #atributos
        self.theta = None
        self.theta_zero = None
        self.cost_history = None
        self.cost_history = None

    def _regular_fit(self, dataset: Dataset) -> 'LogisticRegression':
        """
        fit old
        Gradient Descent atual (implementado no fit inicial)

        :param
        dataset: Dataset
            O dataset para se adaptar ao modelo
        """
        m, n = dataset.shape()

        # inicializar os parâmetros do modelo
        '''inicializar o teta:
            tamanho da variavel theta -> nº de features
            o teta é que vai dar o peso no modelo todo daquelas features. 
            Ou seja, vai estabelecer a regressão linear entre aquela feature e o que vamos prever no final
            '''
        self.theta = np.zeros(n)  # a ponderação de cada feature num modelo linear é 0
        self.theta_zero = 0

        #cost history -> dicionário
        self.cost_history = {}

        # implementação do gradient descent
        # for loop para um maximo de iterações
        for i in range(self.max_iter):
            # estimar os valores de y ( y=mx+b)
            y_pred = np.dot(dataset.X, self.theta) + self.theta_zero

            # calcula o gradiente e atualiza com a taxa de aprendizagem (alpha)
            '''np.dot quando passamos um array de 1 dimensão e um array de 2 dimensões faz o somatório 
            como está na formula
            alpha * 1/m -> multiplicar a dividir pelo numero de amostra (normalização do alpha para o tamanho do datset)'''
            gradient = (self.alpha * (1 / m)) * np.dot(y_pred - dataset.y, dataset.X)

            # calcula o termo de penalização l2
            penalization_term = self.alpha * (self.l2_penalty / m) * self.theta

            # atualiza os parametros (theta, theta_zero) do modelo
            '''
            theta anterior - theta atual (descer) - termo de penalização 

            theta 0 -> não se multiplica por x porque a derivada do b é 0 (logo não se inclui)
            tem de se fazer o somatório das diferenças
            substrair com o antigo theta e atualizar com a taxa de aprendizem 
            '''
            self.theta = self.theta - gradient - penalization_term

            self.theta_zero = self.theta_zero - (self.alpha * (1 / m)) * np.sum(y_pred - dataset.y)

            ##computa a função de custo (self.cost dataset) e armazena o resultado no dicionário cost_history
            self.cost_history[i]=self.cost(dataset)
            # cost_history history(i -1) – cost_history (i)
            # o critério de paragem deve ser uma diferença inferior a 0.0001.
            if i > 0 and (self.cost_history[i-1] - self.cost_history[i]) < 0.0001:
                break

    def _adaptive_fit(self, dataset: Dataset) -> 'LogisticRegression':
        """
        fit new
        semelhante ao método fit mas deve conter o novo algoritmo Gradient DescentDescent.

        :param
        dataset: Dataset
            O dataset para se adaptar ao modelo
        """
        m, n = dataset.shape()

        # inicializar os parâmetros do modelo
        '''inicializar o teta:
            tamanho da variavel theta -> nº de features
            o teta é que vai dar o peso no modelo todo daquelas features. 
            Ou seja, vai estabelecer a regressão linear entre aquela feature e o que vamos prever no final
            '''
        self.theta = np.zeros(n)  # a ponderação de cada feature num modelo linear é 0
        self.theta_zero = 0

        # cost history -> dicionário
        self.cost_history = {}

        # implementação do gradient descent
        # for loop para um maximo de iterações
        for i in range(self.max_iter):
            # estimar os valores de y ( y=mx+b)
            y_pred = np.dot(dataset.X, self.theta) + self.theta_zero

            # calcula o gradiente e atualiza com a taxa de aprendizagem (alpha)
            '''np.dot quando passamos um array de 1 dimensão e um array de 2 dimensões faz o somatório 
            como está na formula
            alpha * 1/m -> multiplicar a dividir pelo numero de amostra (normalização do alpha para o tamanho do datset)'''
            gradient = (self.alpha * (1 / m)) * np.dot(y_pred - dataset.y, dataset.X)

            # calcula o termo de penalização l2
            penalization_term = self.alpha * (self.l2_penalty / m) * self.theta

            # atualiza os parametros (theta, theta_zero) do modelo
            '''
            theta anterior - theta atual (descer) - termo de penalização 

            theta 0 -> não se multiplica por x porque a derivada do b é 0 (logo não se inclui)
            tem de se fazer o somatório das diferenças
            substrair com o antigo theta e atualizar com a taxa de aprendizem 
            '''
            self.theta = self.theta - gradient - penalization_term

            self.theta_zero = self.theta_zero - (self.alpha * (1 / m)) * np.sum(y_pred - dataset.y)

            ##computa a função de custo (self.cost dataset) e armazena o resultado no dicionário cost_history
            self.cost_history[i] = self.cost(dataset)
            # cost_history history(i -1) – cost_history (i)
            # o critério de paragem deve ser uma diferença inferior a 0.0001.
            if i > 0 and (self.cost_history[i - 1] - self.cost_history[i]) < 0.0001:
                #diminuir o valor do alfa: self.alfa= self.alfa/2
                self.alpha = self.alpha / 2

    def fit(self, dataset: Dataset) -> 'RidgeRegression':

        if self.adaptive:
            print('adaptative')
            self._adaptive_fit(dataset)
        else:
            self._regular_fit(dataset)

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
        #Estima os valores de Y usando o theta theta zero e a função sigmoid_function
        predictions = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero)

        #binarizar os dados (classificação binária)
        #Converte os valores estimados em 0 ou 1 (binário). Valores iguais ou superiores a 0.5 tomam o valor de 1.
        # Valores inferiores a 0.5 tomam o valor de 0.
        mask = predictions >= 0.5 #meio de sigmoid
        predictions[mask] = 1
        predictions[~mask] = 0
        return predictions

    def score(self, dataset: Dataset) -> float:
        """
        calcula o erro entre as previsões e os valores reais, usando accuracy

        :param
        dataset: Dataset
            Dataset para calcular a accuracy

        :return
        accuracy: float
            accuracy entre os valores reais e as previsões
        """
        #estima os valores de Y usando o theta e theta _zero
        y_pred = self.predict(dataset)

        #accuracy, para os valores estimados e os valores reais
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
        #Estima os valores de Y usando o theta theta zero e a função sigmoid_function
        predictions = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero)

        #formula cost
        cost = (-dataset.y * np.log(predictions)) - ((1 - dataset.y)* np.log(1 - predictions))
        cost = np.sum(cost) / dataset.shape()[0]
        cost = cost + (self.l2_penalty * np.sum(self.theta**2)/( 2 * dataset.shape()[0]))
        return cost

    def plot_cost_history(self):
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
