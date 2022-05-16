from enum import Enum

recording_bits = 32
recording_sample_rate = 44100


class label_info(Enum):
    START_TIME = "start_time"  # 開始サンプル数
    END_TIME = "end_time"  # 終了サンプル数
    INSTRUMENT = "instrument"  # 楽器番号(MIDI Instrument Number)
    NOTE = "note"  # 音名(MIDI Note Number)
    START_BEAT = "start_beat"  # 開始拍数(最初からの拍数)
    END_BEAT = "end_beat"  # 拍の長さ
    NOTE_VALUE = "note_value"  # 音符の種類

    def header() -> str:
        return "start_time,end_time,instrument,note,start_beat,end_beat,note_value"


# ソロ楽器の楽曲
solo_instrumental_train = [
    2186,
    2241,
    2242,
    2243,
    2244,
    2288,
    2289,
    2659,
    2217,
    2218,
    2219,
    2220,
    2221,
    2222,
    2293,
    2294,
    2295,
    2296,
    2297,
    2202,
    2203,
    2204,
]

solo_instrumental_test = [2191, 2298]


def solo_instrument():
    train_data_paths = list(
        map(
            lambda x: "./resource/musicnet16k/train_data/{}.wav".format(x),
            solo_instrumental_train,
        )
    )
    train_labels_paths = list(
        map(
            lambda x: "./resource/musicnet16k/train_labels/{}.csv".format(x),
            solo_instrumental_train,
        )
    )
    test_data_paths = list(
        map(
            lambda x: "./resource/musicnet16k/test_data/{}.wav".format(x),
            solo_instrumental_test,
        )
    )
    test_labels_paths = list(
        map(
            lambda x: "./resource/musicnet16k/test_labels/{}.csv".format(x),
            solo_instrumental_test,
        )
    )

    return train_data_paths, train_labels_paths, test_data_paths, test_labels_paths
