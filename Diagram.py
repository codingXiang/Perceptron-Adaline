import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Diagram(object):
    def __init__(self , classifier):
        self.classifier = classifier
        self.setup()
    def setup(self):
        self.classifier.train()
    def draw(self , d_type , xlabel = '' , ylabel = '' , legend_loc = '' , title = ''):
        if (d_type == 'classification_line'):
            self.classification_line()
        if (d_type == 'training_process'):
            self.training_progress_diagram()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(loc=legend_loc)
        plt.show()

    def classification_line(self):
        w = self.classifier.w_
        input_data = self.classifier.training_data
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.scatter(input_data[:50 , 0] , input_data[:50 , 1] , color = 'red' , marker = 'o' , label = 'setosa')
        ax1.scatter(input_data[50:100 , 0] , input_data[50:100 , 1] , color = 'blue' , marker = 'x' , label = 'versicolor')
        input_data_min , input_data_max = input_data[: , 0].min() - 1 , input_data[: , 0].max() + 1
        l = np.linspace(input_data_min , input_data_max)
        a , b = -w[1]/w[2], -w[0]/w[2]
        ax1.plot(l, a * l + b, 'b-')

    def training_progress_diagram(self):
        pass

class Perceptron_Diagram(Diagram):
    def __init__(self , classifier):
        super().__init__(classifier)

    def training_progress_diagram(self):
        errors_ = self.classifier.errors_
        plt.plot(range(1 , len(errors_) + 1) , errors_ , marker = 'o')

class Adaline_Diagram(Diagram):
    def __init__(self , classifier):
        super().__init__(classifier)

    def training_progress_diagram(self):
        cost_ = self.classifier.cost_
        plt.plot(range(1 , len(cost_) + 1) , cost_ , marker = 'o')
