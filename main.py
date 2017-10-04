from Algorithm import Perceptron , Adaline
from Data import Dataset

if __name__ == "__main__":
    dataset = Dataset('iris.txt')
    perceptron = Perceptron(dataset , learning_rate = 0.005)
    adaline = Adaline(dataset , learning_rate = 0.001 , n_iter = 100)
    adaline.draw('training_process' ,
              xlabel = 'Epochs' ,
              ylabel = 'Number of misclassifications',
              title = 'Adaline Training Progress',
              legend_loc = 'upper left')
    adaline.draw('classification_line',
              xlabel = 'petal length' ,
              ylabel = 'sepal length',
              title = 'Adaline Linear Separability',
              legend_loc = 'upper left')
    perceptron.draw('training_process' ,
              xlabel = 'Epochs' ,
              ylabel = 'Number of misclassifications',
              title = 'Perceptron Training Progress',
              legend_loc = 'upper left')
    perceptron.draw('classification_line',
              xlabel = 'petal length' ,
              ylabel = 'sepal length',
              title = 'Perceptron Linear Separability',
              legend_loc = 'upper left')
