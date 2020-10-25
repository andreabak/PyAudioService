"""Datatypes and definitions classes for audio related code"""

from __future__ import annotations

import enum
from abc import ABC
from collections import namedtuple
from dataclasses import dataclass
from typing import TypedDict, MutableMapping, Any, IO, Optional

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
"""PCM sample format specifications as used by PCMSampleFormat"""


# pylint: disable=no-member
class PCMSampleFormat(enum.Enum):
    """
    An enum class to represent PCM sample formats.
    """
    float32 = PCMSampleFormatSpec(4, pyaudio.paFloat32, 'f32le', np.float32)  # floating-point 32-bit little-endian
    int32 = PCMSampleFormatSpec(4, pyaudio.paInt32, 's32le', np.int32)        # signed 32-bit little-endian
    int24 = PCMSampleFormatSpec(3, pyaudio.paInt24, 's24le', None)            # signed 24-bit little-endian
    int16 = PCMSampleFormatSpec(2, pyaudio.paInt16, 's16le', np.int16)        # signed 16-bit little-endian
    int8 = PCMSampleFormatSpec(1, pyaudio.paInt8, 's8', np.int8)              # signed 8-bit
    uint8 = PCMSampleFormatSpec(1, pyaudio.paUInt8, 'u8', np.uint8)           # unsigned 8-bit

    @property
    def portaudio(self) -> int:
        """PortAudio (PyAudio) constant value for the PCM sample format enum"""
        return self.value.portaudio_value

    @property
    def ffmpeg(self) -> str:
        """FFmpeg name for the PCM sample format enum"""
        return self.value.ffmpeg_name

    @property
    def numpy(self) -> Optional[np.number]:
        """Numpy numeric type corresponding to the PCM sample format enum.
        Can be None if unsupported (ie. 24-bit)"""
        return self.value.numpy_type

    @property
    def width(self) -> int:
        """Width (size in bytes) for the PCM sample format enum"""
        return self.value.width

    @classmethod
    def get_format_from_width(cls, width: int, unsigned=True) -> PCMSampleFormat:
        """
        Returns the most likely `PCMSampleFormat` enum for the specified width (size in bytes).
        :param width: The desired sample width in bytes (1, 2, 3, or 4)
        :param unsigned: For 1 byte width, specifies whether signed or unsigned format.
        :return: a PCMSampleFormat enum
        """
        if width == 1:
            if unsigned:
                return cls.uint8
            else:
                return cls.int8
        elif unsigned:
            raise ValueError('Unsigned PCM sample format is supported only for 8-bit')
        elif width == 2:
            return cls.int16
        elif width == 3:
            return cls.int24
        elif width == 4:
            return cls.float32
        else:
            raise ValueError(f"Invalid or unsupported PCM sample width: {width} ({width*8}-bit)")

    def __str__(self) -> str:
        """Str representation for the PCM sample format enum"""
        n_type: str = ' floating-point' if self == self.float32 else (' unsigned' if self == self.uint8 else '')
        return f'{self.value.width * 8}-bit{n_type}'


class PyAudioStreamFormatArgs(TypedDict):
    """
    TypedDict that represents PCM format keyword arguments as used by PyAudio
    """
    rate: int
    format: int
    channels: int


class FFmpegFormatArgs(TypedDict):
    """
    TypedDict that represents PCM format commandline arguments for FFmpeg
    """
    ar: int  # Sampling frequency (rate) in Hz
    f: str   # Format
    ac: int  # Audio channels


@dataclass(frozen=True)
class PCMFormat:
    """
    A dataclass to raw PCM format parameters
    """
    rate: int
    """Sampling frequency (rate) in Hz"""

    sample_fmt: PCMSampleFormat
    """Samples format, a PCMSampleFormat enum"""

    channels: int
    """Number of audio channels"""

    @property
    def sample_duration(self) -> float:
        """Duration of a single sample in seconds (= 1 / rate in Hz)"""
        return 1.0 / self.rate

    @property
    def pyaudio_args(self) -> PyAudioStreamFormatArgs:
        """PCM format as PyAudio keyword arguments"""
        return PyAudioStreamFormatArgs(rate=self.rate, format=self.sample_fmt.portaudio, channels=self.channels)

    @property
    def ffmpeg_args(self) -> FFmpegFormatArgs:
        """PCM format as FFmpeg commandline arguments"""
        return FFmpegFormatArgs(ar=self.rate, f=self.sample_fmt.ffmpeg, ac=self.channels)

    @property
    def ffmpeg_args_nofmt(self) -> MutableMapping[str, Any]:
        """PCM format as FFmpeg commandline arguments, excluding samples format"""
        return dict(ar=self.rate, ac=self.channels)

    @property
    def width(self) -> int:
        """Width (size in bytes) of the PCM format (= width of sample_fmt * n. of channels)"""
        return self.sample_fmt.width * self.channels

    def __str__(self) -> str:
        """Str representation for the PCM format"""
        return f'{self.rate}Hz {self.sample_fmt} {self.channels}ch'


class AudioDescriptor(ABC):
    """Abstract base class for audio descriptors (playback call specifiers)"""


@dataclass
class AudioFileDescriptor(AudioDescriptor):
    """Audio descriptor used for playback from a local file path"""
    path: str


@dataclass
class AudioBytesDescriptor(AudioDescriptor, ABC):
    """Abstract base class for audio descriptors with binary data (a bytes string)"""
    audio_data: bytes


@dataclass
class AudioStreamDescriptor(AudioDescriptor, ABC):
    """Abstract base class for audio descriptors with binary stream"""
    audio_stream: IO


@dataclass
class AudioPCMDescriptor(AudioDescriptor, ABC):
    """Abstract base class for audio descriptors with raw PCM audio data"""
    pcm_format: PCMFormat


@dataclass
class AudioEncodedDescriptor(AudioDescriptor, ABC):
    """Abstract base class for audio descriptors with encoded audio data"""
    codec: str


@dataclass
class AudioPCMBytesDescriptor(AudioPCMDescriptor, AudioBytesDescriptor):
    """Audio descriptor used for playback from raw PCM binary data"""


@dataclass
class AudioEncodedBytesDescriptor(AudioEncodedDescriptor, AudioBytesDescriptor):
    """Audio descriptor used for playback from encoded binary data"""


@dataclass
class AudioPCMStreamDescriptor(AudioPCMDescriptor, AudioStreamDescriptor):
    """Audio descriptor used for playback from raw PCM binary stream"""


@dataclass
class AudioEncodedStreamDescriptor(AudioEncodedDescriptor, AudioStreamDescriptor):
    """Audio descriptor used for playback from encoded binary stream"""
