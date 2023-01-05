import numpy as np
import matplotlib.pyplot as plt

from si.data.dataset import Dataset
from si.metrics.mse import mse

class RidgeRegression:
    """
    O RidgeRegression é um modelo linear que utiliza a regularização L2.
    Este modelo resolve o problema da regressão linear utilizando uma técnica de Gradient Descent.

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
    def __init__(self, l2_penalty: float = 1, alpha: float = 0.001, max_iter: int = 1000, adaptive: bool = False):
        """
        :parameter
        l2_penalty: float
            o coeficiente da regularização L2
        alpha: float
            a learning rate (taxa de aprendizagem)
            -> baixinha: permite ao algoritmo não cometer um erro, saltinhos pequeninos
        max_iter: int
            número máximo de iterações
            -> elevado, para com os saltinhos pequeninos chegar ao minimo global
        """
        # parameters
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.adaptive = adaptive

        # attributes
        self.theta = None
        self.theta_zero = None
        self.cost_history = None

    def _regular_fit(self, dataset: Dataset) -> 'RidgeRegression':
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

        # guardar o custo num dicionario
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

            #computa a função de custo (self.cost dataset) e armazena o resultado no dicionário cost_history
            self.cost_history[i] = self.cost(dataset)

            # para obteres o custo da iteração anterior e calcular a diferença da seguinte forma:
            # cost_history history(i -1) – cost_history (i)
            #o critério de paragem deve ser uma diferença inferior a 1
            if i > 1 and self.cost_history[i - 1] - self.cost_history[i] < 1:
                break


    def _adaptive_fit(self, dataset: Dataset) -> 'RidgeRegression':
        """
        fit new
        semelhante ao método fit mas deve conter o novo algoritmo Gradient DescentDescent.
        :param
        dataset: Dataset
            O dataset para se adaptar ao modelo

        :return
        sself: RidgeRegression
            O modelo adaptado
        """
        m, n = dataset.shape()

        #inicializar os parâmetros do modelo
        '''inicializar o teta:
            tamanho da variavel theta -> nº de features
            o teta é que vai dar o peso no modelo todo daquelas features. 
            Ou seja, vai estabelecer a regressão linear entre aquela feature e o que vamos prever no final
            '''
        self.theta = np.zeros(n) #a ponderação de cada feature num modelo linear é 0
        self.theta_zero = 0

        # guardar o custo num dicionario
        self.cost_history = {}

        #implementação do gradient descent
        #for loop para um maximo de iterações
        for i in range(self.max_iter):
            #estimar os valores de y ( y=mx+b)
            y_pred = np.dot(dataset.X, self.theta) + self.theta_zero

            #calcula o gradiente e atualiza com a taxa de aprendizagem (alpha)
            '''np.dot quando passamos um array de 1 dimensão e um array de 2 dimensões faz o somatório 
            como está na formula
            alpha * 1/m -> multiplicar a dividir pelo numero de amostra (normalização do alpha para o tamanho do datset)'''
            gradient = (self.alpha * (1 / m)) * np.dot(y_pred - dataset.y, dataset.X)

            #calcula o termo de penalização l2
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

            # computa a função de custo (self.cost dataset) e armazena o resultado no dicionário cost_history
            self.cost_history[i] = self.cost(dataset)

            #para obteres o custo da iteração anterior e calcular a diferença da seguinte forma:
            # cost_history history(i -1) – cost_history (i)
            # o critério de paragem deve ser uma diferença inferior a 1
            if i > 1 and self.cost_history[i - 1] - self.cost_history[i] < 1:
                ##diminuir o valor do alfa: self.alfa= self.alfa/2
                self.alpha = self.alpha / 2

    def fit(self, dataset: Dataset) -> 'RidgeRegression':

        if self.adaptive:
            print('Adaptive')
            self._adaptive_fit(dataset)
        else:
            self._regular_fit(dataset)

        return self

    def predict(self, dataset: Dataset) -> np.array:
        """
        Prever a saída do conjunto de dados.
        formula -> multiplicar os thetas por X e somar o theta_zero

        (Estima os valores de Y para uma amostra)

        :parameter
        dataset: Dataset
            The dataset to predict the output of

        :return
        predictions: np.array
            The predictions of the dataset
        """
        return np.dot(dataset.X, self.theta) + self.theta_zero

    def score(self, dataset: Dataset) -> float:
        """
        Compute the Mean Square Error of the model on the dataset
        (vai calcular o erro entre as previsões e os valores reais, usando o mse)

        :parameters
        dataset: Dataset
            The dataset to compute the MSE on

        Returns
        -------
        mse: float
            The Mean Square Error of the model
        """
        #Estima os valores de Y usando o theta e theta _zero
        y_pred = self.predict(dataset)

        #mse, para os valores estimados e os valores reais
        return mse(dataset.y, y_pred)

    def cost(self, dataset: Dataset) -> float:
        """
        Compute the cost function (J function) of the model on the dataset using L2 regularization
        (função de custo que usamos no gradiente descent e que permite saber o quão perto o gradient descent está
        do valor de convergencia (que é diferente do mse, são calculados de maneira semelhante mas não é a mesma
        função))

        :parameter
        dataset: Dataset
            The dataset to compute the cost function on

        :return
        cost: float
            The cost function of the model
        """
        #previsões
        y_pred = self.predict(dataset)

        #calcula o J (cost function) entre os valores reais e as previsões
        return (np.sum((y_pred - dataset.y) ** 2) + (self.l2_penalty * np.sum(self.theta ** 2))) / (2 * len(dataset.y))

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

    # make a linear dataset
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    dataset_ = Dataset(X=X, y=y)

    # fit the model
    model = RidgeRegression()
    model.fit(dataset_)

    # get coefs
    print(f"Parameters: {model.theta}")

    # compute the score
    score = model.score(dataset_)
    print(f"Score: {score}")

    # compute the cost
    cost = model.cost(dataset_)
    print(f"Cost: {cost}")

    # predict
    y_pred_ = model.predict(Dataset(X=np.array([[3, 5]])))
    print(f"Predictions: {y_pred_}")
