from model import *
from train import *

def parameter_search(X_train, y_train, X_val, y_val, learning_rates=[0.1, 0.01], hidden_sizes1=[256, 128], hidden_sizes2=[64, 32, 16], weight_decays=[1e-6, 1e-7, 1e-8], num_epochs=200, batch_size=64):
    best_model = None
    best_val_accuracy = 0
    best_hyperparams = {}

    for lr in learning_rates:
        for hidden_size1 in hidden_sizes1:
            for hidden_size2 in hidden_sizes2:
                for weight_decay in weight_decays:
                    # 构建模型
                    model = NeuralNetwork(input_size=784, hidden_size1=hidden_size1, hidden_size2=hidden_size2, output_size=10, activation_func='relu')

                    # 训练模型
                    hyperparams = {"num_epochs": num_epochs, "batch_size": batch_size, "weight_decay": weight_decay, "learning_rate": lr, "hidden_size1": hidden_size1, "hidden_size2": hidden_size2}
                    val_accuracy = train_neural_network(model, hyperparams, X_train, y_train, X_val, y_val)

                    # 模型训练信息
                    print(f"Learning Rate: {lr}, Hidden Size1: {hidden_size1}, Hidden Size2: {hidden_size2}, weight_decay: {weight_decay}, Validation Accuracy: {val_accuracy}")

                    # 更新最佳模型
                    if val_accuracy > best_val_accuracy:
                        best_val_accuracy = val_accuracy
                        best_model = model
                        best_model_weights = model.weights
                        best_hyperparams = hyperparams
    # 保存模型
    model_params = {"input_size": best_model.input_size, "hidden_size1": best_model.hidden_size1, "hidden_size2": best_model.hidden_size2, "output_size": best_model.output_size, "activation_func": best_model.activation_func}
    model_dict = {"params": model_params, "weights": best_model_weights, "hyperparams": best_hyperparams}
    print(best_hyperparams)
    path = f'./weight/params_best.pkl'
    with open(path, 'wb') as f:
        pickle.dump(model_dict, f)
    return best_model, best_hyperparams
