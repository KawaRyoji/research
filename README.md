# 機械学習プログラム

作者: 180441035 川凌司
更新日: 2022/2/15

## 使用方法

```python
    from machine_learning.interfaces import Imachine_learning
    from machine_learning.model import hyper_params
    from machine_learning.k_fold_cross_validation import k_fold_cross_validation

    class some_model(Imachine_learning):
        def create_model(self) -> keras.Model:
            # ここにニューラルネットワークのモデルを記述します
            # モデルはコンパイルを行った状態にしてください
            model = create_some_model()
            return model

        def _create_dataset_process(self, data_path: str, label_path: str) -> tuple:
            # ここにデータファイルパスとラベルファイルパスからデータとラベルを読み込み、
            # データとラベルを対応させてそれぞれ配列に格納してください

            data = load_data(data_path)
            label = load_label(label_path)

            datas = some_process_data(data)
            hotvectors = some_process_label(label) # ラベルは(ワン)ホットベクトルを想定しています

            return datas, hotvectors
    
    # バッチサイズなどのパラメータを設定してください
    params = hyper_params(
        batch_size = 32,
        epochs = 100,
        learning_rate = 0.0002
    )

    # データのディレクトリパスと結果の保存先ディレクトリパスを指定してください
    model = some_model.from_dir(
        "to_train_data_dir_path",
        "to_train_labels_dir_path",
        "to_test_data_dir_path",
        "to_test_labels_dir_path",
        "to_output_dir_path",
    )

    # k分割交差検証ではk_fold_cross_validationを使用してください
    kcv = k_fold_cross_validation(model, params, k=5)
    # hold_out法ではHold_outを使用してください
    # ho = Hold_out(model, params, validation_split=0.1)

    kcv.crate_train_set() # 学習データセットの構築
    kcv.crate_test_set() # テストデータセットの構築

    kcv.train()
    kcv.test()
    kcv.box_plot_history() # テスト結果を箱ひげ図でプロットします
    kcv.plot_average_history() # 学習と検証結果をfoldで平均した結果をプロットします
```

## モジュール

### audio

wavファイルの読みこみやスペクトログラム作成で使用しました。
スペクトログラムはlibrosaで自分で作成したほうが速いかもしれません。

### machine_learning

機械学習関連のプログラム群です。
ここのプログラムにはプログラム上でドキュメントを残しているので参考にしてください。
(ついていないものもありますがご容赦ください)

### musicnet

musicnetデータセット用のプログラム群です。
musicnetを使うという場合にお使いください。

### util

便利関数群です。
fig.pyは使えると思うのでグラフを作る際に使用してください。

## 使用の際に注意すること

機械学習のプログラムですが、結果の保存先を変更しないと実験のたびに結果が上書きされます。
学習データセットなどはtrain()を呼び出すときに指定できるので、
タイムスタンプなどで結果の保存先を変えると良いと思います。
このプログラムを使用するときは、モジュールのディレクトリと同じ階層にプログラムを呼び出すmainファイルを作ると使用しやすいと思います。

このプログラムはかなり荒削りで作ったので、プログラムは見にくいかと思いますが、使用方法で示したようにやれば使えるかなと思います。
ただし、自分は学習データが1次元の場合でしか行っていないので、2次元以上の学習データの入力に対してはおそらくバグがあります。
その場合はすみませんが、自身で書き換えてください。
