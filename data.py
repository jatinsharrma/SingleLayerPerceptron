import pandas as pd
import numpy as np
from scipy.io import arff

from neuron import *

class Data:
    """
    Class to load data from .arff/csv files and store them in a Pandas DataFrame object, which can be accessed by the user through self._data
    """
    def __init__(self):
        self.data = None
        self.train = None
        self.test = None
        self.classes = 0
        self.feature = 0

    def readData(self,path):
        """
        This function takes a file path and save it to self.data

        Args:
            path (string): file path
        """
        if "csv" in path:
            with open(path,"r") as csv:
                df = pd.read_csv(csv)
            self.data = df
        if "arff" in path:
            data, _ = arff.loadarff(path)
            self.data = pd.DataFrame(data)

    def normalizeData(self):
        """
        Normalizes the dataset by subtracting mean from all values
        """
        min = self.data.min(axis=0)
        max = self.data.max(axis=0)
        self.data =  (self.data - min)/(max-min)

    def randomizeData(self):
        """
        Randomize rows of dataframe.
        """
        self.data = self.data.sample(frac=1).reset_index(drop=True)
    
    def splitData(self,ratio=0.8):
        """
        Splits the loaded data into training set and testing set according to ratio passed as argument.
        Args:
            ratio (float, optional): ratio into which dataset need to be split. Defaults to 0.8.

        Returns:
            tuple: tuple of test and train data.
        """
        if not self.train:
            mask = np.random.rand(len(self.data)) < ratio
            self.train = self.data[mask]
            self.train = (
                self.train[self.features].to_numpy().tolist(),
                self.train[self.train.columns[-1]].to_numpy().tolist()
                )
            self.test = self.data[~mask]
            self.test = (
                self.test[self.features].to_numpy().tolist(),
                self.test[self.test.columns[-1]].to_numpy().tolist()
                )
        return (self.train, self.test)
    
    def extractDetails(self):
        """
        Extracts details about the dataset such as number of features/classes etc..
        """
        self.classes = self.data[self.data.columns[-1]].unique().tolist()
        self.features = self.data.columns[:-1].tolist()
    
    def perProcessing(self):
        """
        Performs preprocessing steps on the given dataset like normalizing it or splitting it in two parts for cross validation purpose.
        """
        self.randomizeData()
        self.extractDetails()
        self.splitData()

