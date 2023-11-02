from data import *

class NueralNetwork:
    """
    This class is used to create a neural network. 
    """
    def __init__(self) -> None:
        self.perceptrons = []
        self.accuracy = 0

    def train(self,data : tuple, classes : int):
        """
        Trains the Neural Network using backpropagation algorithm on given dataset and number of classes.
        Args:
            data (tuple): Training data
            classes (int): count of number of classes in datast.
        """
        if classes == 2:
            perceptron = Perceptron(0.9,600)
            perceptron.train(data[0],data[1])
            self.perceptrons.append(perceptron)

    def test(self, data: tuple):
        """
        Tests trained model with testing set.
        Args:
            data (tuple): testing data
        """
        correct = 0
        for perceptron in self.perceptrons:
            for i in range(len(data[1])):
                prediction = perceptron.predict(data[0][i])
                if prediction == data[1][i]:
                    correct += 1
        self.accuracy = correct/len(data[1])
        return correct


if __name__ == "__main__":
    data = Data()
    data.readData("phishing_website.csv")
    data.perProcessing()

    nn = NueralNetwork()
    nn.train(data.train,len(data.train))
    nn.test(data.test)
    print(nn.accuracy)