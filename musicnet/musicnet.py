from enum import Enum

recording_bits = 32
recording_sample_rate = 44100

class label_info(Enum):
    START_TIME = 'start_time' # 開始サンプル数
    END_TIME   = 'end_time'   # 終了サンプル数
    INSTRUMENT = 'instrument' # 楽器番号(MIDI Instrument Number)
    NOTE       = 'note'       # 音名(MIDI Note Number)
    START_BEAT = 'start_beat' # 開始拍数(最初からの拍数)
    END_BEAT   = 'end_beat'   # 拍の長さ
    NOTE_VALUE = 'note_value' # 音符の種類

    def header() -> str:
        return "start_time,end_time,instrument,note,start_beat,end_beat,note_value"