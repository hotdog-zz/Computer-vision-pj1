import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

class SGD:
    # SGD梯度更新
    def __init__(self, learning_rate=0.01, weight_decay=0.01):
        self.learning_rate = learning_rate
        self.weight_deacy = weight_decay

    def update(self, weights, grads):
        # 权重更新
        key = list(weights.keys())
        for weight_name in key:
            grad_name = "d" + weight_name.lower()
            weights[weight_name] -= self.learning_rate * grads[grad_name] + self.weight_deacy * weights[weight_name]

class CrossEntropyLoss:
    # crossentropy损失函数
    def __init__(self):
        pass
    
    def forward(self, x, y_true):
        m = y_true.shape[0]
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        self.y_pred = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        y_pred = np.clip(self.y_pred, 0.000001, 1)
        log_likelihood = -np.log(y_pred[range(m), y_true])
        loss = np.sum(log_likelihood) / m
        return loss
    
    def backward(self, y_true):
        m = y_true.shape[0]
        grad = self.y_pred.copy()
        grad[range(m), y_true] -= 1
        grad /= m
        return grad

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    y_pred = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    return y_pred

def train_neural_network(model, hyperparams, X_train, y_train, X_val, y_val):
    criterion = CrossEntropyLoss()
    sgd = SGD(learning_rate=hyperparams["learning_rate"], weight_decay=hyperparams["weight_decay"])

    num_epochs = hyperparams["num_epochs"]
    batch_size = hyperparams["batch_size"]

    best_val_acc = float('-inf')
    best_weights = {} # 最佳权重
    epoch_loss = [] # 训练集损失
    val_acc = [] # 验证正确率
    val_loss = [] # 验证集损失
    num_batches = len(X_train) // batch_size
    for epoch in tqdm(range(num_epochs), desc='Epochs'):
        # 打乱训练数据
        permutation = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[permutation]
        y_train_shuffled = y_train[permutation]
        batch_loss = []
        # 划分mini-batch训练
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]

            # 前向计算loss
            output = model.forward(X_batch)
            # 加入正则化
            loss = criterion.forward(output, y_batch)
            key = list(model.weights.keys())
            for weight_name in key:
                loss += 0.5 * hyperparams["weight_decay"] * np.sum(np.square(model.weights[weight_name]))
            batch_loss.append(loss)
            # 反向传播计算梯度
            grad = criterion.backward(y_batch)
            model.backward(dz3 = grad)
            # 更新权重
            sgd.update(model.weights, model.gradient)

        epoch_loss.append(np.mean(batch_loss))

        # 计算验证集损失与正确率
        output = model.forward(X_val)
        loss = criterion.forward(output, y_val)
        for weight_name in key:
            loss += 0.5 * hyperparams["weight_decay"] * np.sum(np.square(model.weights[weight_name]))
        val_loss.append(loss)
        y_pred = softmax(output)
        predicted_labels = np.argmax(y_pred, axis=1)
        acc = np.mean(predicted_labels == y_val)
        val_acc.append(acc)
        # 更新最佳模型
        if acc > best_val_acc:
            best_val_acc = acc
            best_weights = model.weights

        # 学习率衰减
        sgd.learning_rate *= 0.95

        # Print training progress
        # print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {epoch_loss[epoch]}, Val Acc = {val_acc[epoch]}")
    # 训练过程可视化
    path = f'./figure/valacc_{int(hyperparams["learning_rate"]*100)}_{int(hyperparams["weight_decay"]*10000000)}_{hyperparams["hidden_size1"]}_{hyperparams["hidden_size2"]}.png'
    epochs = range(1, len(val_acc) + 1)
    plt.clf()
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.title('val acc')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.savefig(path)

    path = f'./figure/valloss_{int(hyperparams["learning_rate"]*100)}_{int(hyperparams["weight_decay"]*10000000)}_{hyperparams["hidden_size1"]}_{hyperparams["hidden_size2"]}.png'
    epochs = range(1, len(val_loss) + 1)
    plt.clf()
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('val loss')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.savefig(path)

    path = f'./figure/trainloss_{int(hyperparams["learning_rate"]*100)}_{int(hyperparams["weight_decay"]*10000000)}_{hyperparams["hidden_size1"]}_{hyperparams["hidden_size2"]}.png'
    epochs = range(1, len(epoch_loss) + 1)
    plt.clf()
    plt.plot(epochs, epoch_loss, label='Train loss')
    plt.title('train loss')
    plt.xlabel('Epochs')
    plt.ylabel('Train loss')
    plt.legend()
    plt.savefig(path)
    # 最佳模型保存
    model_params = {"input_size": model.input_size, "hidden_size1": model.hidden_size1, "hidden_size2": model.hidden_size2, "output_size": model.output_size, "activation_func": model.activation_func}
    model_dict = {"params": model_params, "weights": best_weights, "hyperparams": hyperparams}  
    path = f'./weight/params_{int(hyperparams["learning_rate"]*100)}_{int(hyperparams["weight_decay"]*10000000)}_{hyperparams["hidden_size1"]}_{hyperparams["hidden_size2"]}.pkl'
    with open(path, 'wb') as f:
        pickle.dump(model_dict, f)
    
    return best_val_acc