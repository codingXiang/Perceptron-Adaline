import random
import numpy as np
import pandas as pd
from Diagram import Diagram , Perceptron_Diagram , Adaline_Diagram

class Activate(object):
    def __init__(self,  method):
        self.method = method;
    def process(self , x):
        m = self.method
        if (m == 'sign'):
            return np.where(x >= 0.0 , 1 , -1)
        elif (m == 'step'):
            return np.where(x >= 0.5 , 1 , 0)
        elif (m == 'sigmoid'):
            return 1 / (1 + math.exp(-x))
        else:
            return x
class Algorithm(object):
    def __init__(self , dataset , learning_rate = 0.05 , n_iter = 50):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.training_data = dataset.training_data
        self.output = dataset.output
        self.diagram = Diagram(self)
    def draw(self , d_type , xlabel = '' , ylabel = '' , legend_loc = '' , title = ''):
        self.diagram.draw(d_type , xlabel , ylabel , legend_loc , title )

class Adaline(Algorithm):
    def __init__(self , dataset , learning_rate = 0.05 , n_iter = 50):
        super().__init__(dataset , learning_rate , n_iter)
        self.training_data = self.standard_deviation(dataset.training_data)
        self.diagram = Adaline_Diagram(self)

    # 資料標準化
    def standard_deviation(self , X):
        X_std = np.copy(X)
        X_std[: , 0] = (X[: , 0] - X[: , 0].mean()) / X[: , 0].std()
        X_std[: , 1] = (X[: , 1] - X[: , 1].mean()) / X[: , 1].std()
        return X_std

    def train(self):
        self.w_ = [round(random.uniform(0 , 1) , 2) for x in range(0 , len(self.training_data[0]) + 1)]
        self.cost_ = []
        for _ in range(self.n_iter):
            output = self.net_input(self.training_data)
            errors = (self.output - output)
            self.w_[1:] += self.learning_rate * self.training_data.T.dot(errors)
            self.w_[0] += self.learning_rate * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
            if (cost <= 3):
                break
        return self

    def net_input(self , data):
        return np.dot(data, self.w_[1:]) + self.w_[0]

    def predict(self , method , data):
        activate = Activate(method)
        return activate.process(self.net_input(data))

class Perceptron(Algorithm):
    def __init__(self , dataset , learning_rate = 0.05 , n_iter = 50):
        super().__init__(dataset , learning_rate , n_iter)
        self.diagram = Perceptron_Diagram(self)

    def train(self):
        self.w_ = [round(random.uniform(0 , 1) , 2) for x in range(0 , len(self.training_data[0]) + 1)]
        self.errors_ = []
        while True:
            errors = 0
            count = 0
            for xi , target in zip(self.training_data , self.output):
                update = self.learning_rate * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                if (update != 0):
                    count = 0
                    errors = errors + 1
                else:
                    count = count + 1
            self.errors_.append(errors)
            if (count >= 100):
                break
        return self
    def net_input(self , data):
        return np.dot(data, self.w_[1:]) + self.w_[0]

    def predict(self , data):
        activate = Activate('sign')
        return activate.process(self.net_input(data))
