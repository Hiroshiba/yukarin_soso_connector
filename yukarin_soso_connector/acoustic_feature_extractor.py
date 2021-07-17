from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Sequence, Union

import numpy


@dataclass
class SamplingData:
    array: numpy.ndarray  # shape: (N, ?)
    rate: float

    def resample(self, sampling_rate: float, index: int = 0, length: int = None):
        if length is None:
            length = int(len(self.array) / self.rate * sampling_rate)
        indexes = (numpy.random.rand() + index + numpy.arange(length)) * (
            self.rate / sampling_rate
        )
        return self.array[indexes.astype(int)]

    def split(
        self,
        keypoint_seconds: Union[Sequence[float], numpy.ndarray],
    ):
        keypoint_seconds = numpy.array(keypoint_seconds)
        indexes = (keypoint_seconds * self.rate).astype(numpy.int32)
        arrays = numpy.split(self.array, indexes)
        return [self.__class__(array=array, rate=self.rate) for array in arrays]

    def estimate_padding_value(self):
        values = numpy.concatenate((self.array[:5], self.array[-5:]), axis=0)
        assert len(values) > 0

        value = values[0]
        for i in range(1, len(values)):
            assert numpy.all(value == values[i])

        return value[numpy.newaxis]

    @staticmethod
    def padding(datas: Sequence["SamplingData"], padding_value: numpy.ndarray):
        datas = deepcopy(datas)

        max_length = max(len(d.array) for d in datas)
        for data in datas:
            padding_array = padding_value.repeat(max_length - len(data.array), axis=0)
            data.array = numpy.concatenate([data.array, padding_array])

        return datas

    def all_same(self):
        value = self.array[0][numpy.newaxis]
        return numpy.all(value == self.array)

    @staticmethod
    def collect(
        datas: Sequence["SamplingData"], rate: int, mode: str, error_time_length: float
    ):
        arrays: Sequence[numpy.ndarray] = [
            d.resample(
                sampling_rate=rate, index=0, length=int(len(d.array) * rate / d.rate)
            )
            for d in datas
        ]

        # assert that nearly length
        max_length = max(len(a) for a in arrays)
        for i, a in enumerate(arrays):
            assert (
                abs((max_length - len(a)) / rate) <= error_time_length
            ), f"{i}: {max_length / rate}, {len(a) / rate}"

        if mode == "min":
            min_length = min(len(a) for a in arrays)
            array = numpy.concatenate([a[:min_length] for a in arrays], axis=1).astype(
                numpy.float32
            )

        elif mode == "max":
            arrays = [
                (
                    numpy.pad(a, ((0, max_length - len(a)), (0, 0)))
                    if len(a) < max_length
                    else a
                )
                for a in arrays
            ]
            array = numpy.concatenate(arrays, axis=1).astype(numpy.float32)

        elif mode == "first":
            first_length = len(arrays[0])
            arrays = [
                (
                    numpy.pad(a, ((0, first_length - len(a)), (0, 0)))
                    if len(a) < first_length
                    else a
                )
                for a in arrays
            ]
            array = numpy.concatenate(
                [a[:first_length] for a in arrays], axis=1
            ).astype(numpy.float32)

        else:
            raise ValueError(mode)

        return array

    @classmethod
    def load(cls, path: Path):
        d: Dict = numpy.load(str(path), allow_pickle=True).item()
        array, rate = d["array"], d["rate"]

        if array.ndim == 1:
            array = array[:, numpy.newaxis]

        return cls(array=array, rate=rate)

    def save(self, path: Path):
        numpy.save(str(path), dict(array=self.array, rate=self.rate))


class BasePhoneme(object):
    phoneme_list: Sequence[str]
    num_phoneme: int
    space_phoneme: str

    def __init__(
        self,
        phoneme: str,
        start: float,
        end: float,
    ):
        self.phoneme = phoneme
        self.start = numpy.round(start, decimals=2)
        self.end = numpy.round(end, decimals=2)

    def __repr__(self):
        return f"Phoneme(phoneme='{self.phoneme}', start={self.start}, end={self.end})"

    def __eq__(self, o: object):
        return isinstance(o, BasePhoneme) and (
            self.phoneme == o.phoneme and self.start == o.start and self.end == o.end
        )

    def verify(self):
        assert self.phoneme in self.phoneme_list, f"{self.phoneme} is not defined."

    @property
    def phoneme_id(self):
        return self.phoneme_list.index(self.phoneme)

    @property
    def duration(self):
        return self.end - self.start

    @property
    def onehot(self):
        array = numpy.zeros(self.num_phoneme, dtype=bool)
        array[self.phoneme_id] = True
        return array

    @classmethod
    def parse(cls, s: str):
        """
        >>> BasePhoneme.parse('1.7425000 1.9125000 o:')
        Phoneme(phoneme='o:', start=1.74, end=1.91)
        """
        words = s.split()
        return cls(
            start=float(words[0]),
            end=float(words[1]),
            phoneme=words[2],
        )

    @classmethod
    @abstractmethod
    def convert(cls, phonemes: List["BasePhoneme"]) -> List["BasePhoneme"]:
        pass

    @classmethod
    def load_julius_list(cls, path: Path):
        phonemes = [cls.parse(s) for s in path.read_text().split("\n") if len(s) > 0]
        phonemes = cls.convert(phonemes)

        for phoneme in phonemes:
            phoneme.verify()
        return phonemes

    @classmethod
    def save_julius_list(cls, phonemes: List["BasePhoneme"], path: Path):
        text = "\n".join(
            [
                f"{numpy.round(p.start, decimals=2):.2f}\t"
                f"{numpy.round(p.end, decimals=2):.2f}\t"
                f"{p.phoneme}"
                for p in phonemes
            ]
        )
        path.write_text(text)


class SegKitPhoneme(BasePhoneme):
    phoneme_list = (
        "a",
        "i",
        "u",
        "e",
        "o",
        "a:",
        "i:",
        "u:",
        "e:",
        "o:",
        "N",
        "w",
        "y",
        "j",
        "my",
        "ky",
        "dy",
        "by",
        "gy",
        "ny",
        "hy",
        "ry",
        "py",
        "p",
        "t",
        "k",
        "ts",
        "ch",
        "b",
        "d",
        "g",
        "z",
        "m",
        "n",
        "s",
        "sh",
        "h",
        "f",
        "r",
        "q",
        "sp",
    )
    num_phoneme = len(phoneme_list)
    space_phoneme = "sp"

    @classmethod
    def convert(cls, phonemes: List["SegKitPhoneme"]):
        if "sil" in phonemes[0].phoneme:
            phonemes[0].phoneme = cls.space_phoneme
        if "sil" in phonemes[-1].phoneme:
            phonemes[-1].phoneme = cls.space_phoneme
        return phonemes


class JvsPhoneme(BasePhoneme):
    phoneme_list = (
        "pau",
        "I",
        "N",
        "U",
        "a",
        "b",
        "by",
        "ch",
        "cl",
        "d",
        "dy",
        "e",
        "f",
        "g",
        "gy",
        "h",
        "hy",
        "i",
        "j",
        "k",
        "ky",
        "m",
        "my",
        "n",
        "ny",
        "o",
        "p",
        "py",
        "r",
        "ry",
        "s",
        "sh",
        "t",
        "ts",
        "u",
        "v",
        "w",
        "y",
        "z",
    )
    num_phoneme = len(phoneme_list)
    space_phoneme = "pau"

    @classmethod
    def convert(cls, phonemes: List["JvsPhoneme"]):
        if "sil" in phonemes[0].phoneme:
            phonemes[0].phoneme = cls.space_phoneme
        if "sil" in phonemes[-1].phoneme:
            phonemes[-1].phoneme = cls.space_phoneme
        return phonemes


class OjtPhoneme(BasePhoneme):
    phoneme_list = (
        "pau",
        "A",
        "E",
        "I",
        "N",
        "O",
        "U",
        "a",
        "b",
        "by",
        "ch",
        "cl",
        "d",
        "dy",
        "e",
        "f",
        "g",
        "gw",
        "gy",
        "h",
        "hy",
        "i",
        "j",
        "k",
        "kw",
        "ky",
        "m",
        "my",
        "n",
        "ny",
        "o",
        "p",
        "py",
        "r",
        "ry",
        "s",
        "sh",
        "t",
        "ts",
        "ty",
        "u",
        "v",
        "w",
        "y",
        "z",
    )
    num_phoneme = len(phoneme_list)
    space_phoneme = "pau"

    @classmethod
    def convert(cls, phonemes: List["OjtPhoneme"]):
        if "sil" in phonemes[0].phoneme:
            phonemes[0].phoneme = cls.space_phoneme
        if "sil" in phonemes[-1].phoneme:
            phonemes[-1].phoneme = cls.space_phoneme
        return phonemes


class KiritanPhoneme(BasePhoneme):
    phoneme_list = (
        "pau",
        "a",
        "b",
        "ch",
        "cl",
        "d",
        "e",
        "f",
        "g",
        "gy",
        "h",
        "hy",
        "i",
        "j",
        "k",
        "ky",
        "m",
        "my",
        "n",
        "N",
        "ny",
        "o",
        "p",
        "py",
        "r",
        "ry",
        "s",
        "sh",
        "t",
        "ts",
        "u",
        "v",
        "w",
        "y",
        "z",
    )

    num_phoneme = len(phoneme_list)
    space_phoneme = "pau"

    @classmethod
    def convert(cls, phonemes: List["KiritanPhoneme"]):
        for phoneme in phonemes:
            if phoneme.phoneme == "br":
                phoneme.phoneme = cls.space_phoneme
        return phonemes


class PhonemeType(str, Enum):
    seg_kit = "seg_kit"
    jvs = "jvs"
    kiritan = "kiritan"
    openjtalk = "openjtalk"


phoneme_type_to_class = {
    PhonemeType.seg_kit: SegKitPhoneme,
    PhonemeType.jvs: JvsPhoneme,
    PhonemeType.kiritan: KiritanPhoneme,
    PhonemeType.openjtalk: OjtPhoneme,
}
