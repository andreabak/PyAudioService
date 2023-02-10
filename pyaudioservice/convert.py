from __future__ import annotations

import audioop  # TODO: replace for future compatibility with Py3.13
from io import RawIOBase
from typing import Any, Optional, IO

import numpy as np

from .datatypes import PCMFormat, PCMSampleFormat


__all__ = [
    "float32_to_int32",
    "int32_to_float32",
    "FormatConverter",
    "ResampleStreamReader",
]


def float32_to_int32(data: bytes) -> bytes:
    """
    Convert binary audio data in float32 [-1.0, 1.0] format to int32 (max int32 range).

    :param data: binary audio data in float32 format
    :return: binary audio data in int32 format
    """
    float_values = np.frombuffer(data, dtype=np.float32)
    int_values = (float_values * np.iinfo(np.int32).max).astype(np.int32)
    data = int_values.tobytes()
    return data


def int32_to_float32(data: bytes) -> bytes:
    """
    Convert binary audio data in int32 (max int32 range) to float32 [-1.0, 1.0] format.

    :param data: binary audio data in int32 format
    :return: binary audio data in float32 format
    """
    int_values = np.frombuffer(data, dtype=np.int32)
    float_values = (int_values / np.iinfo(np.int32).max).astype(np.float32)
    data = float_values.tobytes()
    return data


class FormatConverter:
    """
    Class used to convert a raw PCM audio stream from one format to another.
    Use one instance only for one continuous audio stream.

    :param source_format: a `PCMFormat` object specifying the source PCM format
    :param dest_format: a `PCMFormat` object specifying the destination PCM format
    """

    def __init__(self, source_format: PCMFormat, dest_format: PCMFormat):
        if any(f.channels not in (1, 2) for f in (source_format, dest_format)):
            raise ValueError("Only integer stereo or mono PCM samples are supported")
        self.source_format: PCMFormat = source_format
        self.dest_format: PCMFormat = dest_format
        self._ratecv_state: Any = None  # Unknown type

    def convert(self, fragment: bytes) -> bytes:
        """
        Convert an audio fragment

        :param fragment: the source audio fragment in bytes
        :return: the converted audio fragment in bytes
        :raise ValueError: if the length of the source fragment is not a multiple
            of source_format.width
        """
        if len(fragment) % self.source_format.width != 0:
            raise ValueError(
                'Length of audio fragment must be a multiple of "source_format.width"'
            )

        # TODO: endianness handling

        if self.source_format.sample_fmt == PCMSampleFormat.float32:
            fragment = float32_to_int32(fragment)

        if self.source_format.sample_fmt.width != self.dest_format.sample_fmt.width:
            fragment = audioop.lin2lin(
                fragment,
                self.source_format.sample_fmt.width,
                self.dest_format.sample_fmt.width,
            )

        if self.source_format.rate != self.dest_format.rate:
            fragment, self._ratecv_state = audioop.ratecv(
                fragment,
                self.dest_format.sample_fmt.width,
                self.source_format.channels,
                self.source_format.rate,
                self.dest_format.rate,
                self._ratecv_state,
            )

        if self.source_format.channels != self.dest_format.channels:
            if any(
                ch_num not in (1, 2)
                for ch_num in (self.source_format.channels, self.dest_format.channels)
            ):
                raise ValueError(
                    "Only stereo-to-mono or mono-to-stereo conversions are supported"
                )
            if self.source_format.channels == 2:
                fragment = audioop.tomono(
                    fragment, self.dest_format.sample_fmt.width, 1.0, 1.0
                )
            elif self.source_format.channels == 1:
                fragment = audioop.tostereo(
                    fragment, self.dest_format.sample_fmt.width, 1.0, 1.0
                )

        if self.dest_format.sample_fmt == PCMSampleFormat.uint8:
            fragment = audioop.bias(fragment, 1, 128)
        elif self.dest_format.sample_fmt == PCMSampleFormat.float32:
            fragment = int32_to_float32(fragment)
        return fragment


class ResampleStreamReader(RawIOBase, IO[bytes]):
    """
    A IO reader class that reads from a PCM stream and resamples it on the fly
    to a different format.

    :param source_stream: a binary stream to read from
    :param source_format: a `PCMFormat` object specifying the source PCM format
    :param dest_format: a `PCMFormat` object specifying the destination PCM format
    """

    def __init__(
        self,
        source_stream: IO[bytes],
        source_format: PCMFormat,
        dest_format: PCMFormat,
    ):
        self._source_stream: IO[bytes] = source_stream
        self._format_converter: FormatConverter = FormatConverter(
            source_format, dest_format
        )
        self._pending_data: bytes = b""

    @property
    def source_stream(self) -> IO[bytes]:
        """The wrapped unresampled source stream."""
        return self._source_stream

    def read(self, size: Optional[int] = -1) -> bytes:
        """
        Read and convert audio data from the source stream.

        :param size: the number of bytes to read from the source stream.
            These might not match the size of the converted bytes returned.
            If negative, omitted, or None, read until EOF is reached.
        :return: the converted audio data in bytes
        """
        data: bytes = self._pending_data + self._source_stream.read(size)

        excess_size: int = len(data) % self._format_converter.source_format.width
        if excess_size:
            data, self._pending_data = data[:-excess_size], data[-excess_size:]
        else:
            self._pending_data = b""

        return self._format_converter.convert(data)

    def close(self) -> None:
        """Close the stream. Will also close the source stream."""
        self._source_stream.close()

    @property
    def closed(self) -> bool:
        return self._source_stream.closed

    def readable(self):
        return True

    def writable(self):
        return False

    def seekable(self):
        return False
