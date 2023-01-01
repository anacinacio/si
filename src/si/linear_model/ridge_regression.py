import numpy as np
import matplotlib.pyplot as plt

from si.data.dataset import Dataset
from si.metrics.mse import mse
from si.statistics.sigmoid_function import sigmoid_function
from si.metrics.accuracy import accuracy


class RidgeRegression:
    """
    The RidgeRegression is a linear model using the L2 regularization.
    This model solves the linear regression problem using an adapted Gradient Descent technique

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
        # parameters
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter

        # attributes
        self.theta = None
        self.theta_zero = None
        self.cost_history = None

    def fit_new(self, dataset: Dataset) -> 'RidgeRegression':
        m, n = dataset.shape()

        # initialize the model parameters
        self.theta = np.zeros(n)
        self.theta_zero = 0
        self.cost_history = {}

        # gradient descent
        for i in range(self.max_iter):
            # predicted y
            y_pred = np.dot(dataset.X, self.theta) + self.theta_zero

            # computing and updating the gradient with the learning rate
            gradient = (self.alpha * (1 / m)) * np.dot(y_pred - dataset.y, dataset.X)

            # computing the penalty
            penalization_term = self.alpha * (self.l2_penalty / m) * self.theta

            # updating the model parameters
            self.theta = self.theta - gradient - penalization_term
            self.theta_zero = self.theta_zero - (self.alpha * (1 / m)) * np.sum(y_pred - dataset.y)

            # cost
            self.cost_history[i] = self.cost(dataset)
            if i > 1 and (self.cost_history[i-1] - self.cost_history[i]) < 1:
                self.alpha = self.alpha / 2


    def fit_old(selfself, dataset: Dataset) -> 'RidgeRegression':
        """
        Adaptar o modelo ao dataset

        :param
        dataset: Dataset
            O dataset para se adaptar ao modelo

        :return
        sself: RidgeRegression
            O modelo adaptado
        """
        m, n = dataset.shape()

        # initialize the model parameters
        self.theta = np.zeros(n)
        self.theta_zero = 0

        # gradient descent
        for i in range(self.max_iter):
            # predicted y
            y_pred = np.dot(dataset.X, self.theta) + self.theta_zero

            # computing and updating the gradient with the learning rate
            gradient = (self.alpha * (1 / m)) * np.dot(y_pred - dataset.y, dataset.X)

            # computing the penalty
            penalization_term = self.alpha * (self.l2_penalty / m) * self.theta

            # updating the model parameters
            self.theta = self.theta - gradient - penalization_term
            self.theta_zero = self.theta_zero - (self.alpha * (1 / m)) * np.sum(y_pred - dataset.y)
            self.cost_history[i] = self.cost(dataset)

            if i > 1 and self.cost_history[i - 1] - self.cost_history[i] < 1:
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
        Predict the output of the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the output of

        Returns
        -------
        predictions: np.array
            The predictions of the dataset
        """
        return np.dot(dataset.X, self.theta) + self.theta_zero

    def score(self, dataset: Dataset) -> float:
        """
        Compute the Mean Square Error of the model on the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the MSE on

        Returns
        -------
        mse: float
            The Mean Square Error of the model
        """
        y_pred = self.predict(dataset)
        return mse(dataset.y, y_pred)

    def cost(self, dataset: Dataset) -> float:
        """
        Compute the cost function (J function) of the model on the dataset using L2 regularization

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the cost function on

        Returns
        -------
        cost: float
            The cost function of the model
        """
        y_pred = self.predict(dataset)
        return (np.sum((y_pred - dataset.y) ** 2) + (self.l2_penalty * np.sum(self.theta ** 2))) / (2 * len(dataset.y))

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
