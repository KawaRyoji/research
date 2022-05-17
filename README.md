# 機械学習プログラム

作者: 名城大学情報工学科 坂野研究室 川凌司

更新日: 2022/3/27

## はじめに

このプログラムは自身の研究に用いたものです。
研究室で機械学習をやる人に向けて公開します。
自身の研究に合わせて作成している部分があるので、
他のモデルやデータセットで動作するかは保証できません。

## 使用方法

以下に機械学習をk分割交差検証で行うプログラムを示す。
someがついた関数、クラスは自身で定義してください

```python
import keras
from experiments.kcv_experiment import kcv_experiment
from machine_learning.dataset import dataset
from machine_learning.parameter import hyper_params
from machine_learning.model import learning_model

# モデルに入力するデータとそれに対応するラベル(ホットベクトル表現)を返す関数を定義する
def some_construct_process(data_path, label_path) -> tuple:
    data = some_load_data(data_path) # パスから読み込んだ入力データ
    label = some_load_label(label_path) # パスから読み込んだ入力データに対応するラベル(ホットベクトル表現)

    return data, label # データとラベルの組を返す

class some_model(learning_model):
    def create_model(self, **kwargs) -> keras.Model:
        # モデルの定義を記述しモデルを返す
        model = some_model_define()
        
        return model

# 学習セットとテストセットのディレクトリを指定
train_data_dir = "path/to/train/data"
train_label_dir = "path/to/train/label"
test_data_dir = "path/to/test/data"
test_label_dir = "path/to/test/label"

# 学習セットの定義
train_set = dataset.from_dir(
    train_data_dir,
    train_label_dir,
    some_construct_process,
)

# テストセットの定義
test_set = dataset.from_dir(
    test_data_dir,
    test_label_dir,
    some_construct_process,
)

experimental_result_dir = "path/to/experimental/result/directory" # 結果を保存するディレクトリへのパス
params = hyper_params(32, 16, epoch_size=500, learning_rate=0.0001) # 学習時に使用するパラメータ

model = some_model() # 学習させるモデルの作成

# k分割交差検証の準備(この場合はk=5)
ex = kcv_experiment(
    model=model,
    train_set=train_set,
    test_set=test_set,
    params=params,
    k=5,
    experimantal_result_dir=experimental_result_dir,
)

ex.prepare_dataset() # データセットを作成する
ex.train() # モデルを学習し、その結果をプロットする
ex.test() # 学習したモデルの重みでテストし、その結果をプロットする
```

## モジュール

### audio

wavファイルの読みこみやスペクトログラム作成で使用しました。
スペクトログラムはlibrosaで自分で作成したほうが速いかもしれません。

### experiments

自身の研究の実験に用いたスクリプトが置いてあります。
kcv_experiment.pyはk分割交差検証での実験に用いました。
このkcv_experiment.pyは自身の実験用に書かれたものなので、使う場合は用途に応じて書き換えるか、新たにスクリプトを作成してください。

### machine_learning

機械学習関連のプログラム群です。
基本的にここにあるプログラム群で機械学習とその結果を操作します。

### models

機械学習のモデルの定義が置いてあります。

### musicnet

musicnetデータセット用のプログラム群です。
musicnetを使うという場合に使ってください。

### util

便利関数群です。
fig.pyは自身の実験のグラフを作るときに使用しました。

## 使用の際に注意すること

機械学習のプログラムですが、結果の保存先を変更しないと実験のたびに結果が上書きされます。
タイムスタンプなどで結果の保存先を変えると良いと思います。
このプログラムを使用するときは、実験用スクリプトを書いてそれをimportする形で実行すると良いと思います。

このプログラムはかなり荒削りで作ったので、プログラムは見にくいかと思いますが、使用方法で示したようにやれば使えるかなと思います。
ただし、自分は学習データが1次元の場合でしか行っていないので、2次元以上の学習データの入力に対してはおそらくバグがあります。
その場合はすみませんが、自身で書き換えてください。
