import keras.backend as K
import pandas as pd

# NOTE: 算術平均の影響で誤差を含む
# NOTE: ここで算出したF1値は学習の際の表示用に使い、正しいF1値はcsvに保存する際に計算されます
def F1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return innerF1(p, r)


def innerF1(precision, recall):
    return 2. * ((precision * recall) / (precision + recall + K.epsilon()))


def F1_from_log(history: pd.DataFrame):
    metrics = history.columns.tolist()
    if not "recall" in metrics or not "precision" in metrics:
        return history

    has_valid = "val_precision" in metrics
    for i, data in history.iterrows():
        history.loc[i, "F1"] = innerF1(data["precision"], data["recall"])

        if has_valid:
            history.loc[i, "val_F1"] = innerF1(
                data["val_precision"], data["val_recall"]
            )

    return history

def apply_F1_from_log(history_path: str):
    df = pd.read_csv(history_path, index_col=0)
    df = F1_from_log(df)
    pd.DataFrame.to_csv(df, history_path)