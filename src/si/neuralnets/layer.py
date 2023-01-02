import numpy as np

from si.statistics.sigmoid_function import sigmoid_function

class Dense:
    """
    Uma camada densa é uma camada onde cada neurónio está ligado a todos os neurónios da camada anterior

    :parameter
    input_size:
        O número de entradas que a camada irá receber

    output_size:
        O número de resultados que a camada irá produzir

    :Attributes
    weights: np.ndarray
        Os pesos da camada
    bias: np.ndarray
        O bias da camada
    """

    def __init__(self, input_size: int = None, output_size: int = None):
        """
        Inicializar dense layer

        :param
        input_size: linhas, nº de nodos
        output_size: colunas, nº de nodos
        :attributes
        weight: pesos
        bias: bias
        """
        #parameters
        self.input_size = input_size
        self.output_size = output_size

        self.X = None
        #attributes
        self.weights = np.random.randn(input_size, output_size) * 0.01 #nº de linhas, nº de colunas
        self.bias = np.zeros((1, output_size))

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """

        Realiza uma passagem para a frente da camada utilizando o input dado.
        Retorna uma matriz numérica 2d com forma (1, output_size)
        :param
         X: np.ndarray
            entrada para a camada

        :return:
        output: np.ndarray
            saida da camada
        """
        self.X = input_data

        return np.dot(X, self.weights) + self.bias

    #o output_size tem de ser igual
    #colunas - features
    #multiplicação de matrizes -> o nº de colunas da 1º matriz tem de ser igual ao nº de linhas da 2º

    def backward(self, error: np.ndarray, learning_rate: float = 0.01) -> np.ndarray:
        """
        :param
        error: np.ndarray
        learning_rate: float
        """
        error_to_propagate = np.dot(error, self.weights.T)

        self.weights = self.weights - learning_rate * np.dot(self.X.T, error)
        self.bias = self.bias - learning_rate * np.sum(error, axis=0)

        return error_to_propagate
#___________________________________________________________________________
class SigmoidActivation:
    """

    """
    def __unit__(self):
        self.input_data = None

    def forward(self, input_data) -> np.ndarray:
        """
         It returns the sigmoid activation of the given input
        :param input_data: np.ndarray
        :return:
        """
        self.input_data = input_data
        return sigmoid_function(input_data)

    def backward(self, error: np.ndarray, learning_rate: float):
        sigmoid_derivative = sigmoid_function(self.input_data) * (1 - sigmoid_function(self.input_data))

        error_to_propagate = error * sigmoid_derivative

        return error_to_propagate

#__________________________________________________________________________________________________________
class SoftMaxActivation:
    """
    calcula a probabilidade de ocorrência de cada classe
    """
    def __unit__(self):
        self.input_data = None

    def forward(self, input_data) -> np.ndarray:
        """

        :param
        input_data: np.ndarray
        :return:
        Retorna a probabilidade de cada classe
        """
        #definir exponencial do vetor zi
        zi_exp = np.exp(input_data - np.max(input_data))
        #formula
        formula = zi_exp / np.sum(zi_exp, axis=1, keepdims=True)  # axis=1 means that the sum is done by row
        # if set to True will keep the dimension of the array

        return formula

    def backward(self, input_data: np.ndarray, error: np.ndarray) -> np.ndarray:
        soft_max_derivative = input_data * (1 - input_data)

        error_to_propagate = error * soft_max_derivative

        return error_to_propagate

#__________________________________________________________________________________________________________
class ReLUActivation:
    """
    calcula a relação linear retificada
    """
    def __unit__(self):
        self.input_data = None

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        :param
        input_data: np.ndarray

        :return:
        Retorna a relação linear retificada
        """
        self.input_data = input_data

        # maximum between 0 and the input_data, the 0 is to avoid negative values
        return np.maximum(0, input_data)

    def backward(input_data: np.ndarray, error: np.ndarray) -> np.ndarray:
        """
        Computes the backwards pass of the rectified linear relationship.
        :return: Returns the error of the previous layer.
        """
        re_lu_derivative = np.where(input_data > 0, 1, 0)

        error_to_propagate = error * re_lu_derivative

        return error_to_propagate

#__________________________________________________________________________________________________________
class LinearActivation:
    """

    """
    def __unit__(self):
        pass

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        :param
        input_data: np.ndarray

        :return:
        input_data
        """

        return input_data
