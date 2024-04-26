import numpy as np
import pickle
from model import *

def evaluate(path, X_test, y_test):
    # 加载模型
    with open(path, 'rb') as f:
        loaded_model_dict = pickle.load(f)
    params = loaded_model_dict["params"]
    weights = loaded_model_dict["weights"]
    model = NeuralNetwork(**params)
    model.weights = weights
    # 计算正确率
    y_pred = model.forward(X_test)
    accuracy = np.mean(np.argmax(y_pred, axis=1) == y_test)
    return accuracy
