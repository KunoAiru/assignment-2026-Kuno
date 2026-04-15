import pandas as pd
import argparse

# テストデータのあるpathを指定するargparseの設定
parser = argparse.ArgumentParser()
parser.add_argument("data_path", type=str, help="データのあるファイルパスを指定してください")
path = parser.parse_args().data_path

df_val = pd.read_csv(f"{path}/val.tsv", sep="\t", header=None)
df_train = pd.read_csv(f"{path}/train.tsv", sep="\t", header=None)

df = [df_train, df_val]
text = ["train", "val"]

for i in range(len(df)):
    print(f"{text[i]}データの正解ラベルを確認します")
    for j in range(5):
        assert df[i].iloc[j,0] in [0,1], "正解ラベルが0か1以外の値になっています"
    print(f"{text[i]}データの最初5行の正解ラベルは0か1になっています")
