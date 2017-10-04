import numpy as np
import pandas as pd

class Dataset(object):
    def __init__(self , data_path):
        self.df = pd.read_csv(data_path, header = None)
        self.output = np.where(self.df.iloc[0:100, 4].values == 'Iris-setosa', -1, 1)
        self.training_data = self.df.iloc[0:100, [0 , 2]].values
