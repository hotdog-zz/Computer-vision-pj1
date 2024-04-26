from src.model import *
from src.para_search import *
from src.test import *
from src.train import *
import os
import gzip
import numpy as np
import requests
import argparse

def download_fashion_mnist(save_dir):
    base_url = 'https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/'
    files = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']

    # 创建保存数据集的文件夹
    os.makedirs(save_dir, exist_ok=True)

    # 下载并保存数据集文件
    for file in files:
        file_path = os.path.join(save_dir, file)
        if not os.path.exists(file_path):
            url = base_url + file
            r = requests.get(url)
            with open(file_path, 'wb') as f:
                f.write(r.content)

def load_fashion_mnist(dataset_dir):
    files = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
    datasets = []

    # 读取并解压数据集文件
    for file in files:
        with gzip.open(os.path.join(dataset_dir, file), 'rb') as f:
            if "label" in file:
                datasets.append(np.frombuffer(f.read(), dtype=np.uint8, offset=8))  
            else:
                datasets.append(np.frombuffer(f.read(), dtype=np.uint8, offset=16))

    # 将图像数据转换为 numpy 数组并归一化到 [0, 1] 的范围  
    X_train = datasets[0].reshape(-1, 28*28) / 255.0
    y_train = datasets[1]
    X_test = datasets[2].reshape(-1, 28*28) / 255.0
    y_test = datasets[3]

    return np.array(X_train, dtype="float64"), np.array(y_train, dtype="int64"), np.array(X_test, dtype="float64"), np.array(y_test, dtype="int64")

def train_val_split(X, y, val_ratio=0.1, shuffle=True, random_state=None):
    if shuffle:
        # 打乱图片
        if random_state is not None:
            np.random.seed(random_state)
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
    # 验证集下标
    val_size = int(len(X) * val_ratio)

    X_val = X[:val_size]
    y_val = y[:val_size]

    X_train = X[val_size:]
    y_train = y[val_size:]

    return X_train, X_val, y_train, y_val

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    args = parser.parse_args()
    
    os.makedirs('./figure', exist_ok=True)
    os.makedirs('./weight', exist_ok=True)
    # 下载并保存数据集
    dataset_dir = './fashion_mnist'
    download_fashion_mnist(dataset_dir)
    # 加载数据集
    X_train, y_train, X_test, y_test = load_fashion_mnist(dataset_dir)
    # 划分数据集
    X_train, X_val, y_train, y_val = train_val_split(X_train, y_train, val_ratio=0.1, random_state=42)
    if args.train:
        # 网格化搜索
        parameter_search(X_train, y_train, X_val, y_val, num_epochs=200)
    if args.test:
        # 测试模型
        path = "./weight/params_best.pkl"
        print(evaluate(path, X_test, y_test))
