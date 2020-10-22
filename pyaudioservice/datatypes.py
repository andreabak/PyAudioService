from __future__ import annotations

import enum
from abc import ABC
from collections import namedtuple
from dataclasses import dataclass
from typing import TypedDict, MutableMapping, Any, IO

import numpy as np
import pyaudio


__all__ = [
    'PCMSampleFormat',
    'PCMFormat',
    'AudioDescriptor',
    'AudioFileDescriptor',
    'AudioBytesDescriptor',
    'AudioStreamDescriptor',
    'AudioPCMDescriptor',
    'AudioEncodedDescriptor',
    'AudioPCMBytesDescriptor',
    'AudioEncodedBytesDescriptor',
    'AudioPCMStreamDescriptor',
    'AudioEncodedStreamDescriptor',
]


PCMSampleFormatSpec = namedtuple('PCMSampleFormatSpec', 'width, portaudio_value, ffmpeg_name, numpy_type')


class PCMSampleFormat(enum.Enum):
    """
    An enum class to represent PCM sample formats. Enum names correspond to ffmpeg naming, values to portaudio
    """
    float32 = PCMSampleFormatSpec(4, pyaudio.paFloat32, 'f32le', np.float32)  # floating-point 32-bit little-endian
    int32 = PCMSampleFormatSpec(4, pyaudio.paInt32, 's32le', np.int32)        # signed 32-bit little-endian
    int24 = PCMSampleFormatSpec(3, pyaudio.paInt24, 's24le', None)            # signed 24-bit little-endian
    int16 = PCMSampleFormatSpec(2, pyaudio.paInt16, 's16le', np.int16)        # signed 16-bit little-endian
    int8 = PCMSampleFormatSpec(1, pyaudio.paInt8, 's8', np.int8)              # signed 8-bit
    uint8 = PCMSampleFormatSpec(1, pyaudio.paUInt8, 'u8', np.uint8)           # unsigned 8-bit

    @property
    def portaudio(self) -> int:
        return self.value.portaudio_value

    @property
    def ffmpeg(self) -> str:
        return self.value.ffmpeg_name

    @property
    def numpy(self) -> np.number:
        return self.value.numpy_type

    @property
    def width(self) -> int:
        return self.value.width

    @classmethod
    def get_format_from_width(cls, width, unsigned=True) -> PCMSampleFormat:
        """
        Returns a PCMSampleFormat enum for the specified `width`.
        :param width: The desired sample width in bytes (1, 2, 3, or 4)
        :param unsigned: For 1 byte width, specifies signed or unsigned format.
        :return: a PCMSampleFormat enum
        """
        if width == 1:
            if unsigned:
                return cls.uint8
            else:
                return cls.int8
        elif unsigned:
            raise ValueError(f'Unsigned PCM sample format is supported only for 8-bit')
        elif width == 2:
            return cls.int16
        elif width == 3:
            return cls.int24
        elif width == 4:
            return cls.float32
        else:
            raise ValueError(f"Invalid or unsupported PCM sample width: {width} ({width*8}-bit)")

    def __str__(self) -> str:
        n_type: str = ' floating-point' if self == self.float32 else (' unsigned' if self == self.uint8 else '')
        return f'{self.value.width * 8}-bit{n_type}'


class PyAudioStreamFormatArgs(TypedDict):
    rate: int
    format: int
    channels: int


class FFMpegFormatArgs(TypedDict):
    ar: int  # Sampling frequency (rate) in Hz
    f: str   # Format
    ac: int  # Audio channels


@dataclass(frozen=True)
class PCMFormat:
    rate: int
    sample_fmt: PCMSampleFormat
    channels: int

    @property
    def sample_duration(self) -> float:
        return 1.0 / self.rate

    @property
    def pyaudio_args(self) -> PyAudioStreamFormatArgs:
        return PyAudioStreamFormatArgs(rate=self.rate, format=self.sample_fmt.portaudio, channels=self.channels)

    @property
    def ffmpeg_args(self) -> FFMpegFormatArgs:
        return FFMpegFormatArgs(ar=self.rate, f=self.sample_fmt.ffmpeg, ac=self.channels)

    @property
    def ffmpeg_args_nofmt(self) -> MutableMapping[str, Any]:
        return dict(ar=self.rate, ac=self.channels)

    @property
    def width(self) -> int:
        return self.sample_fmt.width * self.channels

    def __str__(self) -> str:
        return f'{self.rate}Hz {self.sample_fmt} {self.channels}ch'


class AudioDescriptor(ABC):
    pass


@dataclass
class AudioFileDescriptor(AudioDescriptor):
    path: str


@dataclass
class AudioBytesDescriptor(AudioDescriptor, ABC):
    audio_data: bytes


@dataclass
class AudioStreamDescriptor(AudioDescriptor, ABC):
    audio_stream: IO


@dataclass
class AudioPCMDescriptor(AudioDescriptor, ABC):
    pcm_format: PCMFormat


@dataclass
class AudioEncodedDescriptor(AudioDescriptor, ABC):
    codec: str


@dataclass
class AudioPCMBytesDescriptor(AudioPCMDescriptor, AudioBytesDescriptor):
    ...


@dataclass
class AudioEncodedBytesDescriptor(AudioEncodedDescriptor, AudioBytesDescriptor):
    ...


@dataclass
class AudioPCMStreamDescriptor(AudioPCMDescriptor, AudioStreamDescriptor):
    ...


@dataclass
class AudioEncodedStreamDescriptor(AudioEncodedDescriptor, AudioStreamDescriptor):
    ...
