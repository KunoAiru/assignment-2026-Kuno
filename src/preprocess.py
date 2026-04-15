import pandas as pd
import numpy as np
import sklearn
import argparse
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# 出力するpathを指定するargparseの設定
parser = argparse.ArgumentParser()
parser.add_argument("output_path", type=str, help="ファイルを出力するパスを指定してください")
parser.add_argument("--train", type=str, default="train.tsv", help="訓練データのファイル名を指定してください")
parser.add_argument("--val", type=str, default="val.tsv", help="検証データのファイル名を指定してください")
parser.add_argument("--test", type=str, default="test.tsv", help="テストデータのファイル名を指定してください")
path = parser.parse_args().output_path
train_file = parser.parse_args().train
val_file = parser.parse_args().val
test_file = parser.parse_args().test

# 標準化のためのscalerを定義
scaler = StandardScaler()

# データと正解ラベルを合わせた表を作成する関数
def create_table(_X,_y):
    result = np.concatenate(( _y.reshape(-1, 1),_X), axis=1)
    return result

# ndarrayの_yから、setosaを1、それ以外を0とする関数
def transform_label(_y):
    for i in range(len(_y)):
        if int(_y[i]) == 0:
            _y[i] = 1
        else:
            _y[i] = 0
    return _y

# データを標準化する関数
def standardize(_file):
    p = Path(_file)
    assert p.exists(), f"指定されたファイル{p.name}が存在しません"
    X = np.loadtxt(_file, delimiter="\t")
    if p.name == train_file:
        transformed_X = scaler.fit_transform(X[:,1:])
        X = np.concatenate((X[:,0].reshape(-1, 1), transformed_X), axis=1)
    else:
        transformed_X = scaler.transform(X[:,1:])
        X = np.concatenate((X[:,0].reshape(-1, 1), transformed_X), axis=1)
    return X

# main関数
def main():
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
    train_table = create_table(X_train, transform_label(y_train))
    val_table = create_table(X_val, transform_label(y_val))
    test_table = create_table(X_test, transform_label(y_test))

    # tsv形式で出力
    np.savetxt(f"{path}/{train_file}", train_table, delimiter="\t")
    np.savetxt(f"{path}/{val_file}", val_table, delimiter="\t")
    np.savetxt(f"{path}/{test_file}", test_table, delimiter="\t")

    # データを標準化
    train_standardized = standardize(f"{path}/{train_file}")
    val_standardized = standardize(f"{path}/{val_file}")
    test_standardized = standardize(f"{path}/{test_file}")

     # tsv形式で出力
    np.savetxt(f"{path}/{train_file}", train_standardized, delimiter="\t")
    np.savetxt(f"{path}/{val_file}", val_standardized, delimiter="\t")
    np.savetxt(f"{path}/{test_file}", test_standardized, delimiter="\t")   

if __name__ == "__main__":
    main()
