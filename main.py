from Algorithm import Perceptron , Adaline
from Data import Dataset

if __name__ == "__main__":
    dataset = Dataset('iris.txt')
    perceptron = Perceptron(dataset , learning_rate = 0.5)
    adaline = Adaline(dataset , learning_rate = 0.01 , n_iter = 100)
    adaline.draw('training_process' ,
              xlabel = 'Epochs' ,
              ylabel = 'Number of misclassifications',
              title = 'Adaline Training Progress - learning rate = ' + str(adaline.learning_rate),
              legend_loc = 'upper left')
    adaline.draw('classification_line',
              xlabel = 'petal length' ,
              ylabel = 'sepal length',
              title = 'Adaline Linear Separability - learning rate = ' + str(adaline.learning_rate),
              legend_loc = 'upper left')
    perceptron.draw('training_process' ,
              xlabel = 'Epochs' ,
              ylabel = 'Number of misclassifications',
              title = 'Perceptron Training Progress - learning rate = ' + str(perceptron.learning_rate),
              legend_loc = 'upper left')
    perceptron.draw('classification_line',
              xlabel = 'petal length' ,
              ylabel = 'sepal length',
              title = 'Perceptron Linear Separability - learning rate = ' + str(perceptron.learning_rate),
              legend_loc = 'upper left')
