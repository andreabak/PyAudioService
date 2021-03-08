"""Main audio stack module"""

from __future__ import annotations

import asyncio
import audioop
import enum
import logging
import subprocess
import time
import uuid
import wave
from abc import ABC, abstractmethod
from asyncio.subprocess import Process
from contextlib import closing, contextmanager, asynccontextmanager
from dataclasses import dataclass, field as dataclass_field
from functools import partial
from inspect import iscoroutinefunction
from threading import Event, Thread
from typing import (
    Callable,
    TypedDict,
    Optional,
    Tuple,
    List,
    MutableMapping,
    Any,
    Union,
    Awaitable,
    Protocol,
    IO,
    Iterator,
    AsyncContextManager,
    ContextManager,
)

import ffmpeg
import pyaudio

from .common import BackgroundService, chunked
from .datatypes import (
    PCMSampleFormat,
    PCMFormat,
    AudioDescriptor,
    AudioFileDescriptor,
    AudioBytesDescriptor,
    AudioStreamDescriptor,
    AudioPCMDescriptor,
    AudioEncodedDescriptor,
)


__all__ = [
    "CHUNK_FRAMES",
    "DEFAULT_FORMAT",
    "FormatConverter",
    "BufferWriteCallback",
    "BufferReadCallback",
    "StreamBuffer",
    "InputStreamHandler",
    "OutputStreamHandler",
    "AudioDeviceIndexType",
    "AudioDeviceInfo",
    "AudioDevicesInfoType",
    "write_to_async_pipe_sane",
    "read_from_async_pipe_sane",
    "BusListenerCallback",
    "BusListenerHandle",
    "AudioService",
]


CHUNK_FRAMES = 1764  # 40ms @ 44100Hz
"""Default chunk size in number of audio frames (samples per channel)
used for buffering and streaming"""

DEFAULT_FORMAT: PCMFormat = PCMFormat(
    rate=44100, sample_fmt=PCMSampleFormat.int16, channels=1
)
"""Default PCM audio format used by the audio service"""


logger: logging.Logger = logging.getLogger(__name__)


TimestampType = float


class PyAudioStreamTimeInfo(TypedDict):
    """
    TypedDict representing PortAudio's StreamTimeInfo struct
    """

    input_buffer_adc_time: float
    """Internal time of the input ADC buffer. 0 if unavailable"""

    current_time: float
    """Current internal PortAudio stream time"""

    output_buffer_dac_time: float
    """Internal time of the output DAC buffer. 0 if unavailable"""


PyAudioStreamCallbackReturn = Tuple[Optional[bytes], int]
PyAudioStreamCallback = Callable[
    [Optional[bytes], int, PyAudioStreamTimeInfo, int], PyAudioStreamCallbackReturn
]
BufferReadCallback = Callable[[int], bytes]
BufferWriteCallback = Callable[[bytes, PCMFormat], None]


class FormatConverter:
    """
    Class used to convert a raw PCM audio stream from one format to another.
    Use one instance only for one continuous audio stream.
    """

    def __init__(self, source_format: PCMFormat, dest_format: PCMFormat):
        """
        Initialization for `FormatConverter`. Use only for one stream at a time.
        :param source_format: a `PCMFormat` object specifying the source PCM format
        :param dest_format: a `PCMFormat` object specifying the destination PCM format
        """
        if any(
            f.sample_fmt == PCMSampleFormat.float32 or f.channels not in (1, 2)
            for f in (source_format, dest_format)
        ):
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
        return fragment


@dataclass
class StreamBuffer:
    """
    Dataclass for buffers (chunks) of audio data attached to a stream
    """

    buffer_data: bytes
    """The audio data chunk in bytes"""

    start_offset: int
    """The offset of the audio data chunk from the start of the stream (in frames / samples)"""

    stream_handler: StreamHandler
    """The `StreamHandler` instance where the audio data chunk originated from"""

    @property
    def size(self) -> int:
        """Length of the audio data chunk in bytes"""
        return len(self.buffer_data)

    @property
    def pcm_format(self) -> PCMFormat:
        """The `PCMFormat` related to the stream and audio data"""
        return self.stream_handler.pcm_format

    def check_size(self) -> bool:
        """
        Sanity check for audio data that must be a multiple of PCM format width
        :return: True if size is valid else None
        """
        return self.size % self.pcm_format.width == 0

    @property
    def frames(self) -> int:
        """Number of frames (samples per channel) contained in the audio data chunk"""
        return self.size // self.pcm_format.width

    @property
    def duration(self) -> float:
        """Overall duration in seconds of the audio chunk"""
        return self.frames / self.pcm_format.rate

    @property
    def start_time_relative(self) -> float:
        """Start time offset in seconds of the audio chunk from the start of the stream"""
        return self.start_offset / self.pcm_format.rate

    @property
    def start_time(self) -> float:
        """Absolute start time of the audio chunk (usually from a monotonic clock)"""
        return self.stream_handler.start_time + self.start_time_relative

    @property
    def end_time(self) -> float:
        """Absolute end time of the audio chunk (usually from a monotonic clock)"""
        return self.start_time + self.duration


class StreamDirection(enum.Enum):
    """Enum for stream direction specification"""

    INPUT = enum.auto()
    OUTPUT = enum.auto()


class StreamHandler(ABC):
    """
    Abstract base class for stream handlers.

    Objects of this class handle audio streams buffers from/to the audio backend
    and provide an interface for control of the stream.
    """

    def __init__(
        self,
        audio_service: AudioService,
        bus: str,
        pcm_format: Optional[PCMFormat] = None,
    ):
        """
        Initialization for `StreamHandler`
        :param audio_service: a running `AudioService` instance
        :param bus: the audio bus name correlated with this stream
        :param pcm_format: the `PCMFormat` for the stream. If not specified here,
            it must be provided before starting the actual audio stream.
        """
        self._audio_service: AudioService = audio_service
        self._bus: str = bus
        self.pcm_format: Optional[PCMFormat] = pcm_format

        self._done_event: Event = Event()
        self._done_event.clear()
        self._stop_event: Event = Event()
        self._stop_event.clear()
        self.stream_error: Optional[Exception] = None
        self._start_time: Optional[TimestampType] = None
        self._done_frames: int = 0

    @property
    @abstractmethod
    def direction(self) -> StreamDirection:
        """The direction of the stream"""

    @abstractmethod
    def _retrieve_audio_data(
        self,
        in_data: Optional[bytes],
        frame_count: int,
        time_info: PyAudioStreamTimeInfo,
        status: int,
    ) -> bytes:
        """
        Retrieve the audio data chunk (input or output) for/from the backend
        :param in_data: input audio data in bytes. None if the stream is not
            an input stream.
        :param frame_count: number of frames (samples per channel) requested
            by PortAudio for output
        :param time_info: PortAudio's StreamTimeInfo for the current chunk
        :param status: PortAudio's status flags for the stream
        :return: the audio data in bytes for output (if an output stream)
            or the input audio data (same as in_data)
        """

    # noinspection PyUnusedLocal
    def pa_callback(
        self,
        in_data: Optional[bytes],
        frame_count: int,
        time_info: PyAudioStreamTimeInfo,
        status: int,
    ) -> PyAudioStreamCallbackReturn:
        """
        Wrapper function used as callback for the PortAudio's stream.
        :param in_data: input audio data in bytes. None if the stream is not
            an input stream.
        :param frame_count: number of frames (samples per channel) requested
            by PortAudio for output
        :param time_info: PortAudio's StreamTimeInfo for the current chunk
        :param status: PortAudio's status flags for the stream
        :return: the audio data in bytes for output (if an output stream).
            None if the stream is not an output stream.
        """
        if self.pcm_format is None:
            logger.error(
                f"Tried starting audio stream, "
                f"but pcm_format was never specified for {self.__class__.__name__}"
            )
            return None, pyaudio.paAbort
        if self._start_time is None:
            self._start_time = time.monotonic()
        if self._stop_event.is_set():
            return None, pyaudio.paAbort
        try:
            audio_data: bytes = self._retrieve_audio_data(
                in_data=in_data,
                frame_count=frame_count,
                time_info=time_info,
                status=status,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.error(f"Error in {self.__class__.__name__}: {exc}", exc_info=True)
            return None, pyaudio.paAbort
        if not audio_data:
            self._done_event.set()
            return None, pyaudio.paComplete
        stream_buffer: StreamBuffer = StreamBuffer(
            audio_data, start_offset=self._done_frames, stream_handler=self
        )
        if not stream_buffer.check_size():
            logger.debug(
                f"Audio buffer has bad size "
                f"(expected {len(audio_data)} % {self.pcm_format.width})"
            )
        self._audio_service.route_buffer(stream_buffer)
        self._done_frames += len(audio_data) // self.pcm_format.width
        if self.direction == StreamDirection.INPUT:
            return None, pyaudio.paContinue
        elif self.direction == StreamDirection.OUTPUT:
            # If PyAudio gets less frames than requested, it stops the stream.
            assume_complete: bool = stream_buffer.frames < frame_count
            pa_flag: int = pyaudio.paComplete if assume_complete else pyaudio.paContinue
            # Setting events just in case
            if assume_complete:
                self._done_event.set()
            return audio_data, pa_flag
        else:
            raise SyntaxError("We should not be here")

    @property
    def bus(self) -> str:
        """The audio bus name from `AudioService` correlated with this stream"""
        return self._bus

    @property
    def done_event(self) -> Event:
        """Stream Event indicating whether the stream is done"""
        return self._done_event

    @property
    def stop_event(self) -> Event:
        """Stream Event indicating whether to stop the stream"""
        return self._stop_event

    @property
    def success(self) -> bool:
        """True if audio stream is finished and no errors were caught, else False"""
        return self._done_event.is_set() and self.stream_error is None

    def set_error(self, error: Exception) -> None:
        """Store an error related to the stream handler instance"""
        self.stream_error = error

    @property
    def start_time(self) -> TimestampType:
        """Internal stream start time (usually a monotonic clock reference)"""
        if self._start_time is None:
            raise ValueError("Stream has not started yet")
        return self._start_time

    @property
    def done_frames(self) -> int:
        """Total number of audio frames streamed (samples per channel)"""
        return self._done_frames


class InputStreamHandler(StreamHandler):
    """
    Stream handler subclass for input streams
    """

    def __init__(
        self,
        *args,
        write_callback: Optional[BufferWriteCallback] = None,
        bus: Optional[str] = None,
        pcm_format: Optional[PCMFormat] = None,
        **kwargs,
    ):
        """
        Initialization for `InputStreamHandler`

        :param args: additional positional arguments for base `StreamHandler.__init__`
        :param write_callback: an optional callable that is called with a
            new chunk of audio data when available
        :param bus: the audio bus name correlated with this stream. If omitted,
            defaults to `AudioService.BUS_INPUT`
        :param pcm_format: the `PCMFormat` requested for the input stream.
            If not specified here, it defaults to the default PCM format.
        :param kwargs: additional keyword arguments for base `StreamHandler.__init__`
        """
        if bus is None:
            bus = AudioService.BUS_INPUT
        if pcm_format is None:
            pcm_format = DEFAULT_FORMAT
        super().__init__(*args, bus=bus, pcm_format=pcm_format, **kwargs)
        self.write_callback: Optional[BufferWriteCallback] = write_callback

    @property
    def direction(self) -> StreamDirection:
        return StreamDirection.INPUT

    def _retrieve_audio_data(
        self,
        in_data: Optional[bytes],
        frame_count: int,
        time_info: PyAudioStreamTimeInfo,
        status: int,
    ) -> bytes:
        audio_data: bytes = in_data
        if self.write_callback is not None and callable(self.write_callback):
            self.write_callback(audio_data, self.pcm_format)
        return audio_data


class OutputStreamHandler(StreamHandler):
    """
    Stream handler subclass for output streams
    """

    def __init__(
        self,
        *args,
        read_callback: Optional[BufferReadCallback] = None,
        bus: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialization for `OutputStreamHandler`

        :param args: additional positional arguments for base `StreamHandler.__init__`
        :param read_callback: a callable that is called whenever a new chunk
            of audio data is requested for output by the PortAudio backend.
            Must accept one int argument as the number of frames requested
            and must return the audio data chunk to output in bytes.
            If not specified here, it must be provided before starting
            the actual audio stream.
        :param bus: the audio bus name correlated with this stream.
            If omitted, defaults to `AudioService.BUS_INPUT`
        :param kwargs: additional keyword arguments for base `StreamHandler.__init__`
        """
        if bus is None:
            bus = AudioService.BUS_OUTPUT
        super().__init__(*args, bus=bus, **kwargs)
        self.read_callback: Optional[BufferReadCallback] = read_callback

    @property
    def direction(self) -> StreamDirection:
        return StreamDirection.OUTPUT

    def _retrieve_audio_data(
        self,
        in_data: Optional[bytes],
        frame_count: int,
        time_info: PyAudioStreamTimeInfo,
        status: int,
    ) -> bytes:
        if self.read_callback is None:
            raise RuntimeError(
                'Must provide a "read_callback" function in order to output audio'
            )
        audio_data: bytes = self.read_callback(frame_count)
        return audio_data


def write_to_async_pipe_sane(
    process: Process, pipe: Optional[asyncio.StreamWriter], data: bytes
) -> None:
    """
    Write to an async subprocess pipe catching and unifying errors
    :param process: the asyncio subprocess `Process`
    :param pipe: the asyncio subprocess pipe, a `StreamWriter` object
    :param data: the binary data to write
    :raise BrokenPipeError: if couldn't write to the pipe
    """
    if process.returncode is not None or not pipe or pipe.is_closing():
        raise BrokenPipeError("Subprocess pipe is closed")
    try:
        pipe.write(data)
    except (OSError, ValueError) as exc:
        if "Errno 22" in str(exc) or "closed pipe" in str(exc):
            raise BrokenPipeError(str(exc)) from exc
        raise


# noinspection PyUnusedLocal
# pylint: disable=unused-argument
async def read_from_async_pipe_sane(
    process: Process, pipe: Optional[asyncio.StreamReader], n: int
) -> bytes:
    """
    Read from an async subprocess pipe catching and unifying errors
    :param process: the asyncio subprocess `Process`
    :param pipe: the asyncio subprocess pipe, a `StreamWriter` object
    :param n: the number of bytes to read
    :return: the read bytes
    :raise BrokenPipeError: if couldn't read from the pipe
    """
    if not pipe:
        raise BrokenPipeError("Subprocess pipe is closed")
    try:
        return await pipe.read(n)
    except (OSError, ValueError) as exc:
        if "Errno 22" in str(exc) or "closed pipe" in str(exc):
            raise BrokenPipeError(str(exc)) from exc
        raise


# noinspection PyProtectedMember,PyUnresolvedReferences,PyBroadException
# pylint: disable=protected-access, broad-except
def close_protocol_with_transport(protocol: Any, force: bool = False) -> None:
    """
    Try closing an asyncio protocol (pipe, subprocess, etc.) if possible.
    :param protocol: the asyncio protocol to close, can be None
    :param force: whether to try calling known force close methods on the transport
    """
    if protocol is None:
        return
    if hasattr(protocol, "close"):
        protocol.close()
    transport: Optional[asyncio.BaseTransport] = getattr(protocol, "transport", None)
    if transport is None:
        transport = getattr(protocol, "_transport", None)
    if transport is None:
        return
    if force:
        try:
            transport._call_connection_lost(None)
        except Exception:
            try:
                transport._force_close()
            except Exception:  # nosec
                pass
        # if (sock := getattr(transport, '_sock', None)) is not None:
        #     sock.close()
        #     setattr(transport, '_sock', None)
    else:
        try:
            transport.close()
        except Exception:  # nosec
            pass


AudioDeviceIndexType = int


class AudioDeviceInfo(TypedDict):
    """
    TypedDict representing audio device information as returned by PortAudio
    """

    index: AudioDeviceIndexType
    structVersion: int
    name: str
    hostApi: int
    maxInputChannels: int
    maxOutputChannels: int
    defaultLowInputLatency: float
    defaultLowOutputLatency: float
    defaultHighInputLatency: float
    defaultHighOutputLatency: float
    defaultSampleRate: Union[int, float]


AudioDevicesInfoType = MutableMapping[AudioDeviceIndexType, AudioDeviceInfo]


BusListenerCallback = Callable[[StreamBuffer], Union[Awaitable[None], None]]


# pylint: disable=undefined-variable
@dataclass(frozen=True)
class BusListenerHandle:
    """
    Dataclass used to represent a handle for a listener attached to an audio bus
    """

    bus_name: str
    """The bus name on which the listener is attached to"""

    callback: BusListenerCallback
    """The listener callback"""

    uuid: str = dataclass_field(
        default_factory=lambda: uuid.uuid1().hex, compare=True, hash=True
    )
    """An unique identifier for the handle.
    Mostly used for identification, comparison and hashing"""


class PlaybackCallable(Protocol):
    """
    Signature protocol specification for `AudioService` internal stream playback methods
    """

    def __call__(
        self, *args, stream_handler: OutputStreamHandler, **kwargs
    ) -> Awaitable:
        ...


class AudioService(BackgroundService):  # TODO: Improve logging for class
    """
    Class for the main audio service.
    Provides an unified interface to the audio backend and related audio functionality.
    """

    CHUNK_FRAMES: int = CHUNK_FRAMES
    """Default chunk size in number of audio frames (samples per channel)
    used for buffering and streaming"""
    DEFAULT_FORMAT: PCMFormat = DEFAULT_FORMAT
    """Default PCM audio format used by the audio service"""

    BUS_OUTPUT: str = "output"
    """Default name for the output audio bus"""
    BUS_INPUT: str = "input"
    """Default name for the input audio bus"""

    @classmethod
    @contextmanager
    def pyaudio_context(cls) -> ContextManager[pyaudio.PyAudio]:
        """
        Context manager class method that creates a `PyAudio` instance
        and manages termination
        :return: the `PyAudio` instance
        """
        pa_instance: pyaudio.PyAudio = pyaudio.PyAudio()
        try:
            yield pa_instance
        finally:
            pa_instance.terminate()

    @classmethod
    def get_audio_devices_info(cls) -> AudioDevicesInfoType:
        """
        Get audio devices info
        :return: A dictionary of `AudioDeviceInfo`, with device index as keys
        """
        with cls.pyaudio_context() as pa:
            devices_info: AudioDevicesInfoType = {
                i: pa.get_device_info_by_index(i) for i in range(pa.get_device_count())
            }
        return devices_info

    @classmethod
    def get_audio_device_by_name(
        cls, device_name: str
    ) -> Tuple[AudioDeviceIndexType, AudioDeviceInfo]:
        """
        Get an audio device by name
        :param device_name: The exact device name to be matched against
            the known available audio devices
        :return: A tuple of audio device index and `AudioDeviceInfo`
        :raise KeyError: If the no audio device is found by the given name
        """
        for device_index, device_info in cls.get_audio_devices_info().items():
            if device_info["name"] == device_name:
                return device_index, device_info
        raise KeyError(f'Audio device with name "{device_name}" not found')

    @classmethod
    def print_audio_devices(cls) -> None:
        """
        Print a list of input audio devices
        """
        for device_index, device_info in cls.get_audio_devices_info().items():
            input_chn: int = device_info["maxInputChannels"]
            output_chn: int = device_info["maxOutputChannels"]
            specs: List[str] = []
            channels_spec_t: str = "{n} {dir}ch"
            dev_type: str = "DUNNO?"
            if input_chn and output_chn:
                dev_type = "IN+OUT"
                specs += [
                    channels_spec_t.format(n=input_chn, dir="in "),
                    channels_spec_t.format(n=output_chn, dir="out "),
                ]
            elif input_chn:
                dev_type = "INPUT "
                specs.append(channels_spec_t.format(n=input_chn, dir=""))
            elif output_chn:
                dev_type = "OUTPUT"
                specs.append(channels_spec_t.format(n=output_chn, dir=""))
            rate: int = int(float(device_info["defaultSampleRate"]))
            specs.append(f"{rate:5d} Hz")
            name: str = device_info["name"]
            # width = dev.get('')
            print(
                f"Index {device_index:2} {dev_type} ({', '.join(specs)}): " f"{name} "
            )

    def __init__(
        self,
        input_device_index: Optional[int] = None,
        output_device_index: Optional[int] = None,
    ):
        """
        Initialization for `AudioService`
        :param input_device_index: The PortAudio input device index to use for
            input streams. If omitted, the default system's input device is used.
        :param output_device_index: The PortAudio output device index to use for
            output streams. If omitted, the default system's output device is used.
        """
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        super().__init__()
        self.input_device_index: Optional[int] = input_device_index
        self.output_device_index: Optional[int] = output_device_index
        self._pa: Optional[pyaudio.PyAudio] = None
        self._bus_listeners: List[BusListenerHandle] = []

    def _create_thread(self) -> Thread:
        return Thread(name="AudioThread", target=self._audio_thread_main)

    @contextmanager
    def _own_pyaudio_context(self) -> ContextManager:
        """
        Context manager helper method that manages the internal
        `PyAudio` instance creation and termination
        """
        with self.pyaudio_context() as pa_instance:
            self._pa = pa_instance
            try:
                yield
            finally:
                self._pa = None

    def _audio_thread_main(self) -> None:
        """Main entry point function for the audio thread"""
        try:
            task: asyncio.Task = self._loop.create_task(
                self._audio_main_async(), name="AsyncMain"
            )
            self._loop.run_until_complete(task)
        finally:
            logger.debug("Audio thread ended")

    async def _audio_main_async(self) -> None:
        """Main entry point coroutine for the audio service (async)"""
        with self._own_pyaudio_context():
            while True:
                if self._stop_event.is_set():
                    logger.debug(
                        "Awaiting on pending tasks and stopping the event loop"
                    )
                    for task in asyncio.Task.all_tasks(self._loop):
                        if task.done() or task.get_name() == "AsyncMain":
                            continue
                        await task
                    self._loop.stop()
                    break
                await asyncio.sleep(0.1)

    @property
    def running(self) -> bool:
        """Whether the audio service is running"""
        return self._thread.is_alive()

    def ensure_running(self) -> None:
        """
        Ensures that the audio service is running, else raise an exception
        :raise RuntimeError: if the audio thread is not running
        """
        if not self.running:
            raise RuntimeError("Audio thread not running")

    def check_exiting(self, dont_raise: bool = False) -> bool:
        """
        Check if the audio service is exiting
        :return: True if exiting and `dont_raise` is True else False
        :raise SystemExit: if the audio service `_stop_event` is set
            and `dont_raise` is False
        """
        exiting: bool = self._stop_event.is_set()
        if exiting and not dont_raise:
            raise SystemExit
        return exiting

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        """Internal asyncio event loop related to the audio service functionality"""
        return self._loop

    def register_bus_listener(
        self, bus_name: str, callback: BusListenerCallback
    ) -> BusListenerHandle:
        """
        Register a new bus listener
        :param bus_name: the audio bus name on which to register the listener
        :param callback: the callback function for the listener
        :return: the new bus listener handle
        """
        listener_handle: BusListenerHandle = BusListenerHandle(bus_name, callback)
        self._bus_listeners.append(listener_handle)
        return listener_handle

    def unregister_bus_listener(self, listener_handle: BusListenerHandle) -> None:
        """
        Unregister a bus listener
        :param listener_handle: the bus listener handle
        """
        if listener_handle in self._bus_listeners:
            self._bus_listeners.remove(listener_handle)
        else:
            logger.warning(
                f"Attempted to remove unregistered bus listener: {repr(listener_handle)}"
            )

    @contextmanager
    def bus_listener(
        self, bus_name: str, callback: BusListenerCallback
    ) -> ContextManager:
        """
        Context manager utility method to register and unregister a bus listener
        :param bus_name: the audio bus name on which to register the listener
        :param callback: the callback function for the listener
        """
        listener_handle: BusListenerHandle = self.register_bus_listener(
            bus_name, callback
        )
        try:
            yield
        finally:
            self.unregister_bus_listener(listener_handle)

    def route_buffer(self, stream_buffer: StreamBuffer) -> None:
        """
        Route a `StreamBuffer` object to all listeners of its `StreamHandler`'s
        related audio bus.
        :param stream_buffer: the `StreamBuffer` object to route
        """
        if self.check_exiting(dont_raise=True):
            return
        for listener_handle in self._bus_listeners:
            if listener_handle.bus_name == stream_buffer.stream_handler.bus:
                asyncio.run_coroutine_threadsafe(
                    self._route_asap(listener_handle, stream_buffer), self._loop
                )

    @staticmethod
    async def _route_asap(
        listener_handle: BusListenerHandle, stream_buffer: StreamBuffer
    ) -> None:
        """
        Actual internal async coroutine that routes a `StreamBuffer` to a bus listener
        :param listener_handle: the listener handle to which to route the buffer to
        :param stream_buffer: the audio stream buffer object
        """
        if iscoroutinefunction(listener_handle.callback):
            await listener_handle.callback(stream_buffer)
        else:
            listener_handle.callback(stream_buffer)

    async def _pa_stream(
        self, stream_handler: StreamHandler, direction: StreamDirection
    ) -> None:
        """
        Handle a PortAudio stream
        :param stream_handler: the `StreamHandler` object to associate to the stream
        :param direction: the direction of the audio stream
        """
        direction_kwargs: MutableMapping[str, Any]
        if direction == StreamDirection.INPUT:
            direction_kwargs = dict(
                input=True, input_device_index=self.input_device_index
            )
        elif direction == StreamDirection.OUTPUT:
            direction_kwargs = dict(
                output=True, output_device_index=self.output_device_index
            )
        else:
            raise AttributeError(f"Invalid direction for PyAudio stream: {direction}")
        with closing(
            self._pa.open(
                **stream_handler.pcm_format.pyaudio_args,
                frames_per_buffer=self.CHUNK_FRAMES,
                **direction_kwargs,
                stream_callback=stream_handler.pa_callback,
            )
        ) as stream:
            try:
                stream.start_stream()
                try:
                    self.check_exiting()
                    while stream.is_active() and not stream_handler.stop_event.is_set():
                        await asyncio.sleep(0.02)
                finally:
                    stream.stop_stream()
            except SystemExit:
                pass
            except Exception as exc:
                logger.warning(f"Error in PyAudio stream: {exc}", exc_info=True)
                stream_handler.set_error(exc)
                raise
            finally:
                stream_handler.done_event.set()
                stream_handler.stop_event.set()

    async def _pa_acquire(self, stream_handler: InputStreamHandler) -> None:
        """
        Handle an input PortAudio stream
        :param stream_handler: the `InputStreamHandler` object to associate to the stream
        """
        await self._pa_stream(stream_handler, StreamDirection.INPUT)

    @contextmanager
    def audio_input(
        self,
        write_callback: Optional[BufferWriteCallback] = None,
        pcm_format: Optional[PCMFormat] = None,
    ) -> ContextManager[InputStreamHandler]:
        """
        Context manager utility method to start and stop an audio input stream
        :param write_callback: an optional callable that is called with a new
            chunk of audio data when available
        :param pcm_format: the `PCMFormat` requested for the input stream.
            If not specified here, it defaults to the default PCM format.
        :return: yields the `InputStreamHandler` for the input stream
        """
        self.ensure_running()
        stream_handler: InputStreamHandler = InputStreamHandler(
            audio_service=self, write_callback=write_callback, pcm_format=pcm_format
        )
        asyncio.run_coroutine_threadsafe(
            self._pa_acquire(stream_handler=stream_handler), self._loop
        )
        try:
            yield stream_handler
        except SystemExit:
            pass
        except Exception as exc:
            logger.warning(f"Error in input PyAudio stream: {exc}", exc_info=True)
            stream_handler.set_error(exc)
            raise
        finally:
            stream_handler.stop_event.set()

    async def _pa_playback(self, stream_handler: OutputStreamHandler) -> None:
        """
        Handle an output PortAudio stream
        :param stream_handler: the `OutputStreamHandler` object to associate to the stream
        """
        await self._pa_stream(stream_handler, StreamDirection.OUTPUT)

    async def _play_wav(
        self, filepath: str, stream_handler: OutputStreamHandler
    ) -> None:
        """
        Coroutine to play a wave file using the `wave` library
        :param filepath: the path of the .wav file to play
        :param stream_handler: an instantiated `OutputStreamHandler`
            to associate to the output stream
        """
        with wave.open(filepath, "rb") as wavefile:
            pcm_format: PCMFormat = PCMFormat(
                rate=wavefile.getframerate(),
                sample_fmt=PCMSampleFormat.get_format_from_width(
                    wavefile.getsampwidth()
                ),
                channels=wavefile.getnchannels(),
            )
            stream_handler.read_callback = wavefile.readframes
            stream_handler.pcm_format = pcm_format
            await self._pa_playback(stream_handler)

    @asynccontextmanager
    async def ffmpeg_subprocess(
        self,
        ffmpeg_spec: ffmpeg.Stream,
        stdin: Union[int, IO, None] = None,
        stdout: Union[int, IO, None] = None,
        stderr: Union[int, IO, None] = None,
        kill_timeout: Optional[float] = None,
        error_callback: Optional[Callable[[Exception], Any]] = None,
    ) -> AsyncContextManager[Process]:
        """
        Async context manager utility method to wrap a FFmpeg process
        :param ffmpeg_spec: a `Stream` object from ffmpeg-python library
            specifying the runtime options for FFmpeg
        :param stdin: the stdin argument for the subprocess call.
            If omitted, defaults to `subprocess.DEVNULL`
        :param stdout: the stdout argument for the subprocess call.
            If omitted, defaults to `subprocess.DEVNULL`
        :param stderr: the stderr argument for the subprocess call.
            If omitted, defaults to `subprocess.DEVNULL`
        :param kill_timeout: the timeout in seconds to wait for the FFmpeg
            process to end before killing it. If omitted or None, it waits until
            the FFmpeg process ends on its own.
        :param error_callback: a callback to send errors caught during the
            process termination phase
        :return: yields the FFmpeg process
        """
        if stdin is None:
            stdin = subprocess.DEVNULL
        if stdout is None:
            stdout = subprocess.DEVNULL
        if stderr is None:
            stderr = subprocess.DEVNULL
        ffmpeg_exec: str = "ffmpeg"
        ffmpeg_args: List[str] = ffmpeg.get_args(ffmpeg_spec)
        ffmpeg_process: Process = await asyncio.create_subprocess_exec(
            ffmpeg_exec,
            *ffmpeg_args,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        caught_exception: Optional[Exception] = None
        error_msg: str
        try:
            yield ffmpeg_process
        except Exception as exc:
            is_broken_pipe: bool = isinstance(exc, BrokenPipeError)
            if is_broken_pipe:
                error_msg = f"FFmpeg pipes broken: {exc}"
            else:
                error_msg = f"Got exception within FFmpeg context: {exc}"
            logger.debug(error_msg)
            caught_exception = exc
            if not is_broken_pipe:
                raise
        finally:
            logger.debug("FFmpeg terminating")
            ffmpeg_retcode: Optional[int] = None
            did_timeout: bool = False
            try:
                ffmpeg_retcode = await asyncio.wait_for(
                    ffmpeg_process.wait(), kill_timeout
                )
            except asyncio.TimeoutError:
                did_timeout = True
            if did_timeout or ffmpeg_retcode != 0:  # TODO: Retrieve stderr?
                if did_timeout:
                    error_msg = (
                        f"FFmpeg still running after {kill_timeout}s, killing it"
                    )
                else:
                    error_msg = f"FFmpeg exited with return code {ffmpeg_retcode}"
                logger.debug(error_msg)
                if did_timeout:
                    ffmpeg_process.kill()
                if error_callback:
                    error_callback(RuntimeError(error_msg))
            if caught_exception is not None or self.check_exiting(dont_raise=True):
                logger.debug("Force closing subprocess transports")
                close_protocol_with_transport(ffmpeg_process.stdin, force=True)
                close_protocol_with_transport(ffmpeg_process.stdout, force=True)
                close_protocol_with_transport(ffmpeg_process.stderr, force=True)

    @asynccontextmanager
    async def _ffmpeg_decoder(
        self,
        input_args: MutableMapping[str, Any],
        pcm_format: PCMFormat,
        stream_handler: OutputStreamHandler,
        pipe_stdin: bool,
    ) -> AsyncContextManager[Process]:
        """
        Async context manager utility method to wrap a FFmpeg-decoded playback stream
        :param input_args: commandline args for FFmpeg for the input audio to be decoded
        :param pcm_format: the requested PCM output format for the stream
        :param stream_handler: an instantiated `OutputStreamHandler` to associate
            to the output stream
        :param pipe_stdin: True if the input audio is expected to be read from
            FFmpeg's stdin, else False
        :return: yields the FFmpeg subprocess
        """
        ffmpeg_spec: ffmpeg.Stream = ffmpeg.input(**input_args).output(
            "pipe:", **pcm_format.ffmpeg_args
        )
        async with self.ffmpeg_subprocess(
            ffmpeg_spec,
            stdin=subprocess.PIPE if pipe_stdin else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            kill_timeout=2.0,
            error_callback=stream_handler.set_error,
        ) as ffmpeg_process:

            def read_stdout_pipe(frame_count: int) -> bytes:
                nonlocal ffmpeg_process, pcm_format
                # Ensuring we don't read less than what requested, until output has finished
                target_bytes: int = frame_count * pcm_format.width
                buffer: bytes = b""
                while (bytes_to_read := target_bytes - len(buffer)) > 0:
                    if self.check_exiting(
                        dont_raise=True
                    ):  # Break on service termination request
                        break
                    read_coro: Awaitable = read_from_async_pipe_sane(
                        ffmpeg_process, ffmpeg_process.stdout, bytes_to_read
                    )
                    try:
                        read_bytes: bytes = asyncio.run_coroutine_threadsafe(
                            read_coro, self._loop
                        ).result()
                    except BrokenPipeError:
                        break
                    if not read_bytes:
                        if not pipe_stdin or ffmpeg_process.stdin.is_closing():
                            # if ffmpeg_process.stdout and not ffmpeg_process.stdout.at_eof():
                            #     ffmpeg_process.stdout.close()
                            break
                        time.sleep(
                            0.5
                            * (bytes_to_read / pcm_format.width)
                            * pcm_format.sample_duration
                        )
                    buffer += read_bytes
                return buffer

            stream_handler.read_callback = read_stdout_pipe
            stream_handler.pcm_format = pcm_format
            playback_task: asyncio.Task = self._loop.create_task(
                self._pa_playback(stream_handler)
            )

            try:
                yield ffmpeg_process
            except SystemExit:
                pass
            except BrokenPipeError as exc:
                logger.info(f"Force stopped playback: {exc}")
                raise
            finally:
                ffmpeg_retcode: Optional[int]
                while True:
                    ffmpeg_retcode = ffmpeg_process.returncode
                    if playback_task.done() or ffmpeg_retcode not in (None, 0):
                        stream_handler.stop_event.set()
                        break
                    await asyncio.sleep(0.1)
                close_protocol_with_transport(ffmpeg_process.stdout)

    async def _play_ffmpeg_piped(
        self,
        audio: Union[bytes, IO],
        stream_handler: OutputStreamHandler,
        codec: Optional[str] = None,
        data_pcm_format: Optional[PCMFormat] = None,
        pcm_format: Optional[PCMFormat] = None,
    ) -> None:
        """
        Coroutine to play a FFmpeg-decoded stream piped to stdin from a buffer
        or bytes string.
        :param audio: the audio buffer (stream) object or bytes string
        :param stream_handler: an instantiated `OutputStreamHandler` to
            associate to the output stream
        :param codec: the codec used to decode the input audio data, optional
        :param data_pcm_format: the PCM format for the input audio data, optional
        :param pcm_format: the requested PCM output format for the stream,
            if omitted the default PCM format will be used
        """
        if pcm_format is None:
            pcm_format = self.DEFAULT_FORMAT
        input_args: MutableMapping[str, Any] = dict(filename="pipe:")
        if codec is not None:
            input_args["acodec"] = codec
        if data_pcm_format is not None:
            input_args.update(**data_pcm_format.ffmpeg_args)
        async with self._ffmpeg_decoder(
            input_args, pcm_format, stream_handler, pipe_stdin=True
        ) as ffmpeg_process:
            chunk_size: int
            if data_pcm_format is not None:
                chunk_size = (
                    (self.CHUNK_FRAMES * data_pcm_format.rate) // pcm_format.rate
                ) * data_pcm_format.width
            else:
                chunk_size = self.CHUNK_FRAMES
            sleep_time: float = (
                chunk_size / pcm_format.width * pcm_format.sample_duration * 0.5
            )
            # Feed audio data to ffmpeg stdin
            try:
                is_stream: bool = hasattr(audio, "read")
                if not is_stream:
                    if not isinstance(audio, bytes):
                        raise ValueError(
                            f"Only binary IO streams or bytes data are supported, got: {repr(audio)}"
                        )
                    chunks_generator: Iterator[bytes] = chunked(audio, chunk_size)
                while True:
                    input_chunk: bytes
                    if is_stream:  # FIXME: Careful with blocking reads within async
                        input_chunk = audio.read(chunk_size)
                        if not input_chunk:
                            audio.close()
                            break
                    else:
                        try:
                            input_chunk = next(chunks_generator)
                        except StopIteration:
                            break
                    if stream_handler.stop_event.is_set() or self.check_exiting(
                        dont_raise=True
                    ):
                        break
                    write_to_async_pipe_sane(
                        ffmpeg_process, ffmpeg_process.stdin, input_chunk
                    )
                    await asyncio.sleep(sleep_time)
            finally:
                ffmpeg_process.stdin.close()

    async def _play_ffmpeg_file(
        self,
        filepath: str,
        stream_handler: OutputStreamHandler,
        pcm_format: Optional[PCMFormat] = None,
    ) -> None:
        """
        Coroutine to play a FFmpeg-decoded file
        :param filepath: the path of the audio file
        :param stream_handler: an instantiated `OutputStreamHandler` to associate
            to the output stream
        :param pcm_format: the requested PCM output format for the stream,
            if omitted the default PCM format will be used
        """
        if pcm_format is None:
            pcm_format = self.DEFAULT_FORMAT
        input_args: MutableMapping[str, Any] = dict(filename=filepath)
        async with self._ffmpeg_decoder(
            input_args, pcm_format, stream_handler, pipe_stdin=False
        ):
            pass

    def _play_stream(
        self, playback_callable: PlaybackCallable, blocking: bool
    ) -> OutputStreamHandler:
        """
        Internal wrapper method to start a playback stream
        :param playback_callable: a callable that must accept a `OutputStreamHandler`
            and must return the coroutine to schedule to the asyncio loop for playback
        :param blocking: if True, will wait until the stream to end to return
        :return: the `OutputStreamHandler` associated with the playback stream
        """
        self.ensure_running()
        stream_handler: OutputStreamHandler = OutputStreamHandler(audio_service=self)
        asyncio.run_coroutine_threadsafe(
            playback_callable(stream_handler=stream_handler), self._loop
        )
        if blocking:
            while not stream_handler.done_event.is_set():
                time.sleep(0.1)  # Using short sleeps for faster ctrl-c breaking
        return stream_handler

    def play_data(
        self,
        audio: Union[bytes, IO],
        codec: Optional[str] = None,
        data_pcm_format: Optional[PCMFormat] = None,
        blocking: bool = True,
    ) -> OutputStreamHandler:
        """
        Start a playback stream for audio data in a binary stream or bytes string
        :param audio: the audio buffer (stream) object or bytes string
        :param codec: the codec used to decode the input audio data, optional
        :param data_pcm_format: the PCM format for the input audio data, optional
        :param blocking: if True, will wait until the stream to end to return
        :return: the `OutputStreamHandler` associated with the playback stream
        """
        # noinspection PyTypeChecker
        playback_callable: PlaybackCallable = partial(
            self._play_ffmpeg_piped,
            audio=audio,
            codec=codec,
            data_pcm_format=data_pcm_format,
        )
        return self._play_stream(playback_callable, blocking=blocking)

    def play_file(self, filepath: str, blocking: bool = True) -> OutputStreamHandler:
        """
        Start a playback stream for an audio file
        :param filepath: the path of the audio file
        :param blocking: if True, will wait until the stream to end to return
        :return: the `OutputStreamHandler` associated with the playback stream
        """
        # noinspection PyTypeChecker
        playback_callable: PlaybackCallable = partial(
            self._play_ffmpeg_file, filepath=filepath
        )
        return self._play_stream(playback_callable, blocking=blocking)

    def play_descriptor(
        self, audio_descriptor: AudioDescriptor, blocking: bool = True
    ) -> OutputStreamHandler:
        """
        Start a playback stream for the giving audio descriptor, automatically
            picking the most appropriate method
        :param audio_descriptor: the audio descriptor that specifies playback
        :param blocking: if True, will wait until the stream to end to return
        :return: the `OutputStreamHandler` associated with the playback stream
        """
        try:
            if isinstance(audio_descriptor, AudioFileDescriptor):
                return self.play_file(audio_descriptor.path, blocking=blocking)
            elif isinstance(
                audio_descriptor, (AudioBytesDescriptor, AudioStreamDescriptor)
            ):
                audio: Union[bytes, IO]
                if isinstance(audio_descriptor, AudioBytesDescriptor):
                    audio = audio_descriptor.audio_data
                elif isinstance(audio_descriptor, AudioStreamDescriptor):
                    audio = audio_descriptor.audio_stream
                else:
                    raise SyntaxError("wrong class, sanity check failed")
                if isinstance(audio_descriptor, AudioPCMDescriptor):
                    return self.play_data(
                        audio,
                        data_pcm_format=audio_descriptor.pcm_format,
                        blocking=blocking,
                    )
                elif isinstance(audio_descriptor, AudioEncodedDescriptor):
                    return self.play_data(
                        audio, codec=audio_descriptor.codec, blocking=blocking
                    )
                else:
                    raise ValueError("unknown or unhandled audio descriptor type")
            else:
                raise ValueError("unknown or unhandled audio descriptor type")
        except ValueError as exc:
            raise ValueError(
                f"Could not play audio descriptor {repr(audio_descriptor)}: {exc}"
            ) from exc
