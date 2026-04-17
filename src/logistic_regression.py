import torch
import argparse
import pandas as pd
import math

# コマンド引数の設定
parser = argparse.ArgumentParser()
parser.add_argument("data_path", type=str, help="ファイルを入力するパスを指定してください")
parser.add_argument("learning_rate", type=float, help="学習率を指定してください")
parser.add_argument("num_epochs", type=int, help="エポック数を指定してください")
parser.add_argument("--random_seed",type=int,default=0,help="乱数シードを指定してください")
parser.add_argument("--train", type=str, default="train.tsv", help="訓練データのファイル名を指定してください")
path = parser.parse_args().data_path
learning_rate = parser.parse_args().learning_rate
num_epochs = parser.parse_args().num_epochs
random_seed = parser.parse_args().random_seed
train_file = parser.parse_args().train

# sigmoid関数
def sigmoid(z):
    return 1 / (1 + torch.exp(-z))

# 損失関数の勾配を求める関数
def loss_grad(X,y,w):
    return X*(sigmoid(torch.dot(X,w))-y)

def main():
    df_train = pd.read_csv(f"{path}/{train_file}", sep="\t", header=None)
    tensor_train = torch.tensor(df_train.values, dtype=torch.float32)

    # 乱数値を固定
    torch.manual_seed(random_seed)

    y_train = tensor_train[:,0]
    X_train = tensor_train[:,1:]

    w =torch.randn(X_train.shape[1])
    print(f"学習前の重みw:{w}")

    for _ in range(num_epochs):
        for i in range(X_train.shape[0]):
            grad = loss_grad(X_train[i], y_train[i], w)
            w -= learning_rate*grad

    print(f"学習後の重みw:{w}")

if __name__ == "__main__":
    main()