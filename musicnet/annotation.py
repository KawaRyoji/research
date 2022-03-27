from musicnet.musicnet import label_info
from audio import midi
from math import floor
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

from util.calc import point2sec


class dataset_label:
    def __init__(
        self, start_time, end_time, instrument, note, start_beat, end_beat, note_value
    ) -> None:

        self.start_time = start_time
        self.end_time = end_time
        self.instrument = instrument
        self.note = note
        self.start_beat = start_beat
        self.end_beat = end_beat
        self.note_value = note_value

    @classmethod
    def from_structured_array(cls, array):
        return cls(
            array[label_info.START_TIME.value],
            array[label_info.END_TIME.value],
            array[label_info.INSTRUMENT.value],
            array[label_info.NOTE.value],
            array[label_info.START_BEAT.value],
            array[label_info.END_BEAT.value],
            array[label_info.NOTE_VALUE.value],
        )

    def to_sturctured_array(self):
        data = np.zeros(
            len(self.start_time),
            dtype=[
                (label_info.START_TIME.value, "<i4"),
                (label_info.END_TIME.value, "<i4"),
                (label_info.INSTRUMENT.value, "<i4"),
                (label_info.NOTE.value, "<i4"),
                (label_info.START_BEAT.value, "<f8"),
                (label_info.END_BEAT.value, "<f8"),
                (label_info.NOTE_VALUE.value, "<U26"),
            ],
        )

        data[label_info.START_TIME.value] = self.start_time
        data[label_info.END_TIME.value] = self.end_time
        data[label_info.INSTRUMENT.value] = self.instrument
        data[label_info.NOTE.value] = self.note
        data[label_info.START_BEAT.value] = self.start_beat
        data[label_info.END_BEAT.value] = self.end_beat
        data[label_info.NOTE_VALUE.value] = self.note_value

        return data

    def convert_fs(self, origin_fs, target_fs) -> None:
        if origin_fs == target_fs:
            return

        assert origin_fs > target_fs

        fs_ratio = target_fs / origin_fs

        self.start_time = np.ceil(self.start_time * fs_ratio)
        self.end_time = np.ceil(self.end_time * fs_ratio)

    def save(self, filepath):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        labels = self.to_sturctured_array()
        np.savetxt(
            filepath, labels, delimiter=",", fmt="%s", header=label_info.header()
        )

    def frame_mid_pitches(self, fstart_sample, fend_sample):
        mid = floor((fstart_sample + fend_sample) / 2)
        
        if mid < 0:
            raise IndexError()
        
        index = (self.start_time <= mid) & (mid <= self.end_time)
        index = np.array(index)
        res = self.note[index]
        return np.array(res)

    def plot_origin_label(self, fs, max_sec=None):
        plt.figure()
        for i in range(len(self.start_time)):
            plt.hlines(
                y=self.note[i],
                xmin=point2sec(fs, self.start_time[i]),
                xmax=point2sec(fs, self.end_time[i]),
                linewidth=4,
            )

        if max_sec is None:
            max_sec = point2sec(fs, self.end_time[i])

        plt.xlim(0, max_sec)
        plt.ylim(0, midi.num_notes)
        plt.grid()
        plt.show()

    @classmethod
    def plot(cls, labels):
        plt.figure()
        for i, label in enumerate(labels):
            for j, note in enumerate(label):
                if note == 0:
                    continue
                plt.hlines(y=j, xmin=i, xmax=i + 1, linewidth=4)

        plt.ylim(0, midi.num_notes)
        plt.grid()
        plt.show()

    @classmethod
    def load(cls, file_path):
        labels = np.genfromtxt(
            file_path, delimiter=",", names=True, dtype=None, encoding="utf-8"
        )
        return cls.from_structured_array(labels)

    @classmethod
    def list2hotvector(cls, label_list):
        hotvector = np.zeros(midi.num_notes)
        for label in label_list:
            hotvector[label] = 1

        return np.array(hotvector, dtype=np.float32)
