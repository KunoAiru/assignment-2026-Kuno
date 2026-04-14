import matplotlib
import optuna
import torch
import wandb
import seaborn
import transformers
import tqdm
import pandas as pd
import numpy as np
import sklearn
import argparse

# 出力するpathを指定するargparseの設定
parser = argparse.ArgumentParser()
parser.add_argument("output_path", type=str, help="ファイルを出力するパスを指定してください")
path = parser.parse_args().output_path


# データと正解ラベルを合わせた表を作成する関数
def create_table(_X,_y):
    result = np.concatenate(( _y.reshape(-1, 1),_X), axis=1)
    return result

iris = sklearn.datasets.load_iris()
X = iris.data
y = iris.target

# 訓練・検証データとテストデータに分割（8 : 2）
X_train_val,X_test,y_train_val,y_test = sklearn.model_selection.train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 訓練データと検証データに分割（8 : 1）
X_train,X_val,y_train,y_val = sklearn.model_selection.train_test_split(
    X_train_val, y_train_val, test_size=0.125, random_state=42
)

# 正解ラベルとデータを合わせた表を作成
train_table = create_table(X_train, y_train)
val_table = create_table(X_val, y_val)
test_table = create_table(X_test, y_test)

# tsv形式で出力
np.savetxt(f"{path}/train.tsv", train_table, delimiter="\t")
np.savetxt(f"{path}/val.tsv", val_table, delimiter="\t")
np.savetxt(f"{path}/test.tsv", test_table, delimiter="\t")
