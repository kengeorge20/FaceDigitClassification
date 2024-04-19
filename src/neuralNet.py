import numpy as np
from classificationMethod import ClassificationMethod
from util import Counter

class NeuralNetworkClassifier(ClassificationMethod):
    def __init__(self, legalLabels, input_size, hidden_size, output_size, learning_rate=0.01, max_iterations=10):
        super().__init__(legalLabels)
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_iterations = max_iterations

        # Initialize weights
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2. / self.input_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2. / self.hidden_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        for epoch in range(self.max_iterations):
            for x, y in zip(trainingData, trainingLabels):
                x = self._ensure_numpy_array(x)
                y = np.array(y)  # Assuming y is already in a suitable format like one-hot or a list of numbers

                # Forward pass
                z1 = np.dot(x, self.weights_input_hidden) + self.bias_hidden
                a1 = self.sigmoid(z1)
                z2 = np.dot(a1, self.weights_hidden_output) + self.bias_output
                a2 = self.sigmoid(z2)

                # Backward pass
                delta2 = (a2 - y) * self.sigmoid_prime(z2)
                delta1 = np.dot(delta2, self.weights_hidden_output.T) * self.sigmoid_prime(z1)

                # Update weights
                self.weights_hidden_output -= self.learning_rate * np.dot(a1.T, delta2)
                self.weights_input_hidden -= self.learning_rate * np.dot(x.reshape(1, -1).T, delta1)
                self.bias_output -= self.learning_rate * np.sum(delta2, axis=0, keepdims=True)
                self.bias_hidden -= self.learning_rate * np.sum(delta1, axis=0, keepdims=True)

    def classify(self, data):
        data = np.array([self._ensure_numpy_array(d) for d in data])
        z1 = np.dot(data, self.weights_input_hidden) + self.bias_hidden
        a1 = self.sigmoid(z1)
        z2 = np.dot(a1, self.weights_hidden_output) + self.bias_output
        return np.argmax(self.sigmoid(z2), axis=1)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_prime(self, z):
        sig = self.sigmoid(z)
        return sig * (1 - sig)

    def _ensure_numpy_array(self, counter):
        if isinstance(counter, Counter):
            return np.array([counter.get(i, 0) for i in range(self.input_size)])
        return counter