import numpy as np

class Sigmoid:
    # Sigmoid激活函数
    def __init__(self):
        pass
    
    def forward(self, x):
        return 1. / (1 + np.exp(-x))
    
    def backward(self, x):
        s = 1. / (1 + np.exp(-x))
        return s * (1 - s)

class ReLU:
    # ReLU激活函数
    def __init__(self):
        pass
    
    def forward(self, x):
        return np.maximum(0, x)
    
    def backward(self, x):
        return np.where(x > 0, 1, 0)

class Tanh:
    # Tanh激活函数
    def __init__(self):
        pass
    
    def forward(self, x):
        return np.tanh(x)
    
    def backward(self, x):
        return 1 - np.tanh(x) ** 2


class NeuralNetwork:
    def __init__(self, input_size = 784, hidden_size1 = 512, hidden_size2 = 128, output_size = 10, activation_func = "sigmoid", **kwargs):
        # 模型参数
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.activation_func = activation_func
        # 模型权重及初始化
        self.weights = {}
        self.weights["W1"] = np.random.randn(input_size, hidden_size1) / np.sqrt(input_size / 2)
        self.weights["b1"] = np.zeros((1, hidden_size1))
        self.weights["W2"] = np.random.randn(hidden_size1, hidden_size2) / np.sqrt(hidden_size1 / 2)
        self.weights["b2"] = np.zeros((1, hidden_size2))
        self.weights["W3"] = np.random.randn(hidden_size2, output_size) / np.sqrt(hidden_size2 / 2)
        self.weights["b3"] = np.zeros((1, output_size))
        # 模型激活函数
        if activation_func == 'sigmoid':
            self.activation = Sigmoid()
        elif activation_func == 'relu':
            self.activation = ReLU()
        elif activation_func == 'tanh':
            self.activation = Tanh()
        
    def forward(self, X):
        # 前向传播
        self.X = X
        self.z1 = np.dot(X, self.weights["W1"]) + self.weights["b1"]
        self.a1 = self.activation.forward(self.z1)
        self.z2 = np.dot(self.a1, self.weights["W2"]) + self.weights["b2"]
        self.a2 = self.activation.forward(self.z2)
        self.z3 = np.dot(self.a2, self.weights["W3"]) + self.weights["b3"]
        return self.z3
    
    def backward(self, dz3):
        # 反向传播
        dw3 = np.dot(self.a2.T, dz3)
        db3 = np.sum(dz3, axis=0, keepdims=True)
        
        dz2 = np.dot(dz3, self.weights["W3"].T) * self.activation.backward(self.z2)
        dw2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        
        dz1 = np.dot(dz2, self.weights["W2"].T) * self.activation.backward(self.z1)
        dw1 = np.dot(self.X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        # 保存梯度
        self.gradient = {}
        self.gradient["dw3"] = dw3
        self.gradient["db3"] = db3
        self.gradient["dw2"] = dw2
        self.gradient["db2"] = db2
        self.gradient["dw1"] = dw1
        self.gradient["db1"] = db1


