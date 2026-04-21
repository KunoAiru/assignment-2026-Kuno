import os
import argparse
import pandas as pd
import torch
import wandb
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("data_path", type=str, help="ファイルを入力するパスを指定してください")
parser.add_argument("output_path", type=str, help="モデルファイルを出力するパスを指定してください")
parser.add_argument("learning_rate", type=float, help="学習率を指定してください")
parser.add_argument("num_epochs", type=int, help="エポック数を指定してください")
parser.add_argument("--random_seed", type=int, default=0, help="乱数シードを指定してください")
parser.add_argument("--train", type=str, default="train.tsv", help="訓練データのファイル名を指定してください")
parser.add_argument("--val", type=str, default="val.tsv", help="検証データのファイル名を指定してください")
parser.add_argument("--test", type=str, default="test.tsv", help="テストデータのファイル名を指定してください")
args = parser.parse_args()

import_path = args.data_path
output_path = args.output_path
learning_rate = args.learning_rate
num_epochs = args.num_epochs
random_seed = args.random_seed
train_file = args.train
val_file = args.val
test_file = args.test


class EarlyStopping:
    def __init__(self, patience=3, verbose=False):
        self._patience = patience
        self._verbose = verbose
        self._f1score = -1.0
        self._step = 0

    def validate(self, f1score):
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


def predict_y(x, model):
    with torch.no_grad():
        prob = torch.sigmoid(model(x.unsqueeze(0))).squeeze().item()
    return 1 if prob >= 0.5 else 0


def evaluate(X, y, model, early_stopping=None, debug=True):
    TP = TN = FP = FN = 0

    for i in range(X.shape[0]):
        pred_y = predict_y(X[i], model)
        true_y = int(y[i].item())

        if true_y == pred_y:
            if true_y == 1:
                TP += 1
            else:
                TN += 1
        else:
            if true_y == 1:
                FN += 1
            else:
                FP += 1

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

    is_early_stop = early_stopping.validate(f1_score) if early_stopping else False

    if debug:
        print(f"accuracy: {accuracy}, precision: {precision}, recall: {recall}, F1_score: {f1_score}")

    return [accuracy, precision, recall, f1_score, is_early_stop]


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

    model = torch.nn.Linear(X_train.shape[1], 1)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=3, verbose=True)

    wandb.init(project="logistic_regression_autograd",config={
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "random_seed": random_seed,
        },
    )

    for _ in tqdm(range(num_epochs), desc="Epochs"):
        model.train()

        for i in tqdm(range(X_train.shape[0]), desc="Batch", leave=False):
            optimizer.zero_grad()
            outputs = torch.sigmoid(model(X_train[i].unsqueeze(0))) 
            target = y_train[i].view(1, 1)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            wandb.log({"step_train_loss": loss.item()})

        model.eval()
        with torch.no_grad():
            eval_result = evaluate(X_val, y_val.squeeze(), model, early_stopping)

            epoch_train_loss = sum(criterion(torch.sigmoid(model(X_train[i].unsqueeze(0))),y_train[i].view(1, 1)).item()for i in range(X_train.shape[0])) / X_train.shape[0]

            epoch_val_loss = sum(criterion(torch.sigmoid(model(X_val[i].unsqueeze(0))),y_val[i].view(1, 1)).item()for i in range(X_val.shape[0])) / X_val.shape[0]

            wandb.log({
                "epoch_train_loss": epoch_train_loss,
                "epoch_val_loss": epoch_val_loss,
                "accuracy": eval_result[0],
                "precision": eval_result[1],
                "recall": eval_result[2],
                "F1_score": eval_result[3],
            })

            if eval_result[4]:
                break

    model.eval()
    with torch.no_grad():
        test_outputs = torch.sigmoid(model(X_test)).squeeze().tolist()

    test_predictions = [1 if output >= 0.5 else 0 for output in test_outputs]
    test_eval = evaluate(X_test, y_test.squeeze(), model, debug=False)

    test_results_df = pd.DataFrame({
        "predicted": test_predictions,
        "actual": y_test.squeeze().tolist(),
    })

    test_results_df["accuracy"] = test_eval[0]
    test_results_df["precision"] = test_eval[1]
    test_results_df["recall"] = test_eval[2]
    test_results_df["F1_score"] = test_eval[3]

    test_results_df.to_csv(f"{output_path}/test_results.csv", index=False)
    torch.save(model.state_dict(), f"{output_path}/model_autograd.pt")

    wandb.finish()


if __name__ == "__main__":
    main()