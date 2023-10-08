import copy

class Perceptron:
    """
    Perceptron classifier for binary classification problems
    """
    def __init__(self,eta,iterations,bias=1,label=None) -> None:
        self.eta = eta
        self.iterations = iterations
        self.weight_vector = None
        self.bias = bias
        self.label = label

    def train(self,input_pattern,desired_output):
        """
        This function trains the perceptron.
        Args:
            input_pattern (list): List of all patterns
            desired_output (list): List of desired output
        """
        no_features = len(input_pattern[0])+1
        self.weight_vector = [0]*no_features
        pattern_index = 0
        for i in range(self.iterations):
            pattern = copy.deepcopy(input_pattern[pattern_index])
            output = self.predict(pattern)
            error = desired_output[pattern_index] - output
            if error != 0:
                delta_w = self.deltaWeightVector(pattern,error)
                self.updateWeightVector(delta_w)
            pattern_index += 1
            if pattern_index == len(input_pattern):
                pattern_index = 0
    
    def localInducedField(self,input_v):
        """
        This function calculates the local induced field.
        Args:
            input_v (list): input pattern 

        Returns:
            list: Calculated local induced field
        """
        local_induced_field = 0
        for i in range(len(input_v)):
            local_induced_field += input_v[i] * self.weight_vector[i]
        return local_induced_field
    
    def updateWeightVector(self,delta_w):
        """
        This function updates weight vector after each iteration
        Args:
            delta_w (list): delta weight vector
        """
        for i in range(len(self.weight_vector)):
            self.weight_vector[i] += delta_w[i]
    
    def deltaWeightVector(self,pattern,error):
        """
        This function calculates the delta weight vector.
        Args:
            pattern (list): pattern list
            error (int): error 

        Returns:
            list: calculated delta weight vector.
        """
        constant = self.eta * error
        delta_w = []
        for i in range(len(self.weight_vector)):
            delta_w.append(constant * pattern[i])
        return delta_w
    
    def activationFunction(self,vk):
        """
        This is a placeholder method to be overridden by specific network implementation.
        This will calculate the output of neuron based on its inputs and weights.

        Args:
            vk (int): local induced field

        Returns:
            int: activation value
        """
        if vk > 0:
            return 1
        else:
            return -1
        
    def predict(self,pattern):
        """
        This function returns predicted class label using trained perceptron model

        Args:
            pattern (list): pattern user has given

        Returns:
            int: predicted value
        """
        pattern.insert(0,self.bias)
        return self.activationFunction(self.localInducedField(pattern))
