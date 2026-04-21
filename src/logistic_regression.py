import torch
import wandb
from tqdm import tqdm
import argparse
import pandas as pd

# コマンド引数の設定
parser = argparse.ArgumentParser()
parser.add_argument("data_path", type=str, help="ファイルを入力するパスを指定してください")
parser.add_argument("output_path", type=str, help="モデルファイルを出力するパスを指定してください")
parser.add_argument("learning_rate", type=float, help="学習率を指定してください")
parser.add_argument("num_epochs", type=int, help="エポック数を指定してください")
parser.add_argument("--random_seed",type=int,default=0,help="乱数シードを指定してください")
parser.add_argument("--train", type=str, default="train.tsv", help="訓練データのファイル名を指定してください")
parser.add_argument("--val", type=str, default="val.tsv", help="検証データのファイル名を指定してください")
parser.add_argument("--test", type=str, default="test.tsv", help="テストデータのファイル名を指定してください")
parse = parser.parse_args()
import_path = parse.data_path
output_path = parse.output_path
learning_rate = parse.learning_rate
num_epochs = parse.num_epochs
random_seed = parse.random_seed
train_file = parse.train
val_file = parse.val
test_file = parse.test

# sigmoid関数
def sigmoid(z):
    return 1 / (1 + torch.exp(-z))

# 損失関数の勾配を求める関数
def loss_grad(X,y,w):
    return X*(sigmoid(torch.dot(X,w))-y)

class EarlyStopping:
    def __init__(self,patience=3,verbose=False):
        self._patience = patience
        self._verbose = verbose
        self._f1score = -1.0 #F1スコアの最大化問題
        self._step = 0

    def validate(self,f1score):
        if self._f1score >= f1score:
            self._step += 1
            if self._step > self._patience:
                if self._verbose:
                    print("早期終了しました")
                return True
        else:
            self._f1score = f1score
            self._step = 0
        return False

def predict_y(X,w):
    predict = sigmoid(torch.matmul(X,w))
    pre_y = 1 if predict >= 0.5 else 0
    return pre_y

def loss(X,y,w):
    z = torch.dot(X,w)
    return -y*torch.log(sigmoid(z))-(1-y)*torch.log(1-sigmoid(z))

def evaluate(X,y,w,early_stopping=None,debug=True):
    TP = TN = FP = FN = 0
    for i in range(X.shape[0]):
        pred_y = predict_y(X[i],w)
        if int(y[i]) == pred_y:
            if int(y[i]) == 1:
                TP += 1
            else:
                TN += 1
        else:
            if int(y[i]) == 1:
                FN += 1
            else:
                FP += 1

    accuracy = (TP+TN)/(TP+TN+FP+FN)
    precision = TP/(TP+FP) if TP+FP > 0 else 0.0
    recall = TP/(TP+FN) if TP+FN > 0 else 0.0
    F1_score = 2 * (precision*recall)/(precision+recall) if precision+recall > 0 else 0.0
    is_early_stop = early_stopping.validate(F1_score) if early_stopping else False

    eval_list = [accuracy,precision,recall,F1_score,is_early_stop]
    if debug:
        print(f"accuracy: {eval_list[0]}, precision: {eval_list[1]}, recall: {eval_list[2]}, F1_score: {eval_list[3]}")
    return eval_list

def main():

    df_train = pd.read_csv(f"{import_path}/{train_file}", sep="\t", header=None)
    df_val = pd.read_csv(f"{import_path}/{val_file}", sep="\t", header=None)
    df_test = pd.read_csv(f"{import_path}/{test_file}", sep="\t", header=None)
    tensor_train = torch.tensor(df_train.values, dtype=torch.float32)
    tensor_val = torch.tensor(df_val.values, dtype=torch.float32)
    tensor_test = torch.tensor(df_test.values, dtype=torch.float32)

    # 乱数値を固定
    torch.manual_seed(random_seed)

    y_train = tensor_train[:,0]
    X_train = tensor_train[:,1:]
    y_val = tensor_val[:,0]
    X_val = tensor_val[:,1:]
    y_test = tensor_test[:,0]
    X_test = tensor_test[:,1:]

    w =torch.randn(X_train.shape[1])
    early_stopping = EarlyStopping(patience=3,verbose=True)

    wandb.init(project="logistic_regression", config={
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "random_seed": random_seed,
        })

    for _ in tqdm(range(num_epochs), desc="Epochs"):
        for i in tqdm(range(X_train.shape[0]), desc="Batch"):
            wandb.log({"step_train_loss": loss(X_train[i], y_train[i], w).item()})
            grad = loss_grad(X_train[i], y_train[i], w)
            w -= learning_rate*grad
        with torch.no_grad():      
            eval = evaluate(X_val,y_val,w,early_stopping)
            epoch_train_loss = sum(loss(X_train[i], y_train[i], w).item() for i in range(X_train.shape[0])) / X_train.shape[0]
            epoch_val_loss = sum(loss(X_val[i], y_val[i], w).item() for i in range(X_val.shape[0])) / X_val.shape[0]
            wandb.log({"epoch_train_loss": epoch_train_loss, "epoch_val_loss": epoch_val_loss, "accuracy": eval[0], "precision": eval[1], "recall": eval[2], "F1_score": eval[3]})            
            if eval[4]: # 早期終了の判定
                break

    torch.save(w, f"{output_path}/model.pt")

    model = torch.load(f"{output_path}/model.pt")
    with torch.no_grad():
        print("訓練データの出力と評価結果:")
        for i in range(X_train.shape[0]):
            print(f"予測: {predict_y(X_train[i], model)}, 実際: {y_train[i]}")
        evaluate(X_train,y_train,model,debug=True)

        print("検証データの出力と評価結果:")
        for i in range(X_val.shape[0]):
            print(f"予測: {predict_y(X_val[i], model)}, 実際: {y_val[i]}")
        evaluate(X_val,y_val,model,debug=True)

        print("テストデータの出力と評価結果:")
        for i in range(X_test.shape[0]):
            print(f"予測: {predict_y(X_test[i], model)}, 実際: {y_test[i]}")
        evaluate(X_test,y_test,model,debug=True)


if __name__ == "__main__":
    main()