"""Datatypes and definitions classes for audio related code"""

from __future__ import annotations

import enum
import time
from abc import ABC
from collections import namedtuple
from dataclasses import dataclass
from io import BytesIO
from threading import Lock
from typing import TypedDict, MutableMapping, Any, IO, Optional, Iterator

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
    'BufferedAudioData',
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


class BufferedAudioData:
    """
    A class for buffered audio data that can be concurrently written and read.

    Provides a `write()` and `read()` method with separated position pointers to
    allow for a buffer-like producer-consumer pattern.

    Source raw audio bytes data can be added to the buffer by the producer
    function using `write()`, that accepts either bytes or another AudioData
    instance, with matching sample_rate and sample_width.

    Then the consumer function can use the `read()` method to get the written
    data out of the buffer.

    When the producer is done writing data, it should call the `done()`
    function to signal the data is complete.
    The property `.complete` will be true if the data is considered complete.
    Once data is considered "complete", trying to use the `write()` method
    will raise a `BufferError`, while trying to `read()` beyond the buffer
    size will raise an `EOFError`, instead of just returning empty data.

    N.B. In the end the buffer will contain the whole written data,
         backed up by memory, until garbage-collected.
    """
    def __init__(self, pcm_format: PCMFormat, frame_data: bytes = None, complete: bool = False):
        """
        Initialization for `BufferedAudioData`
        :param pcm_format: The PCM raw audio format, specified by an instance of `PCMFormat`
        :param frame_data: Optional initial frame data in bytes, that will be put in the buffer
        :param complete: Optional bool that if True signals that the provided data is considered complete
        """
        self.pcm_format: PCMFormat = pcm_format
        self._data_buffer: BytesIO = BytesIO()
        self._buffer_lock: Lock = Lock()
        self._read_pos: int = 0
        self._complete: bool
        if frame_data is not None:
            self._complete = False
            self.write(frame_data)
        self._complete = complete

    @property
    def complete(self) -> bool:
        """
        A read only property indicating whether the audio data contained
        in the buffer is considered complete or not.
        :return: True if complete else False
        """
        return self._complete

    def done(self) -> None:
        """
        Set the internal flag to signal that the audio data in the buffer
        is complete and no more data is going to be written to it.
        """
        self._complete = True

    def _ensure_sample_width(self, data_size: int, nonraising: bool = False) -> Optional[bool]:
        """
        Ensures the specified data size is a multiple of the PCM format width
        :param data_size: The specified data size
        :param nonraising: Instead of raising an error, return a boolean
        :return: a boolean indicated whether the data size is a multiple of the PCM format width
        :raise IOError: If the data size is not a multiple of the PCM format width
        """
        is_multiple: bool = (data_size % self.pcm_format.width) == 0
        if not nonraising and not is_multiple:
            raise IOError('Data size must be a multiple of "sample_width"')
        return is_multiple

    def write(self, data: bytes) -> None:
        """
        Write raw audio data bytes to the buffer.
        :param data: The raw audio data bytes
        :raise BufferError: If the data is already considered complete
        """
        if self.complete:
            raise BufferError('Buffer data is complete!')
        self._ensure_sample_width(len(data))
        with self._buffer_lock:
            self._data_buffer.write(data)
            self._data_buffer.flush()

    def read(self, size: int = -1) -> bytes:
        """
        Read audio data from the buffer.
        Internally keeps track of the buffer's last read position.
        :param size: The amount of bytes to read. If omitted, all data will be
                     read until the end of the buffer.
        :raise EOFError: If there's no more data to read from the buffer and
                         the data is considered to be complete.
        :return: A bytes string of raw audio data. If no new data is present
                 in the buffer since the last `read()` the returned bytes
                 string will be empty (== b'').
        """
        if size != -1:
            self._ensure_sample_width(size)
        data: bytes
        if self.is_data_available():  # Raises EOFError if no data and self.complete
            with self._buffer_lock:  # Acquire lock while we're seeking cursor around and reading
                write_pos = self._data_buffer.tell()  # Back up the current buffer (write) position
                self._data_buffer.seek(self._read_pos)  # Seek to the last known read position
                data = self._data_buffer.read(size)  # Read data
                self._read_pos = self._data_buffer.tell()  # Save the read position
                self._data_buffer.seek(write_pos)  # Restore the previous buffer (write) position
            # If we're enforcing only writing and reading in `sample_width` chunks, our returned
            # audio data bytes length must be a multiple of that.
            assert self._ensure_sample_width(len(data), nonraising=True), \
                'Data length is not a multiple of sample_width'
        else:
            data = b''  # Signals there's no new data in the buffer, but more might come in the future
        return data

    def is_data_available(self) -> bool:
        """
        Check if any data is available for reading.
        :return: True if data can be read, else False
        :raise EOFError: If there's no more data to read from the buffer and
                         the data is considered to be complete.
        """
        with self._buffer_lock:
            unread_data: bool = self._data_buffer.tell() > self._read_pos
        if not unread_data and self.complete:
            raise EOFError('All data from the buffer has been read and data is complete')
        return unread_data

    def __len__(self) -> int:
        """
        Get the size of the buffer in bytes
        :return: Integer buffer size in bytes
        """
        with self._buffer_lock:
            return self._data_buffer.tell()

    @property
    def seconds(self) -> float:
        """
        Get the length in seconds of the audio data stored in the buffer.
        This is calculated dividing the total buffer size in bytes
        by `sample_width` and `sample_rate`.
        :return: The length in seconds, in float
        """
        return len(self) / self.pcm_format.width / self.pcm_format.rate

    def get_whole(self) -> bytes:
        """
        Get the whole buffer contents
        :return: All the bytes in the buffer
        """
        return self._data_buffer.getvalue()

    def generate(self) -> Iterator[bytes]:
        """
        Generator method that waits for data to be available in the buffer
        and yields it for chunked usage/transmission.

        N.B. Uses a short blocking sleep time when no audio data is available.
        :return: An iterator of bytes containing the raw audio data
        """
        while True:
            try:
                has_data: bool = self.is_data_available()
            except EOFError:  # Audio data stream is complete
                break
            if has_data:
                yield self.read()  # Yield bytes to request to be sent
            else:
                time.sleep(0.05)  # Short sleep, supposing we're doing this in a different thread
