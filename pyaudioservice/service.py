"""Main audio stack module"""

from __future__ import annotations

import asyncio
import audioop
import enum
import subprocess
import time
import uuid
import wave
from abc import ABC, abstractmethod
from contextlib import closing, contextmanager, asynccontextmanager
from dataclasses import dataclass, field as dataclass_field
from functools import partial
from inspect import iscoroutinefunction
from threading import Event, Thread
from typing import Callable, TypedDict, Optional, Tuple, List, MutableMapping, Any, Union, Awaitable, Protocol, \
    IO, Iterator

import ffmpeg
import pyaudio

from .datatypes import PCMSampleFormat, PCMFormat, AudioDescriptor, AudioFileDescriptor, AudioBytesDescriptor, \
    AudioStreamDescriptor, AudioPCMDescriptor, AudioEncodedDescriptor
from ..common import BackgroundService, TimestampType, chunked
from ..logger import custom_log


__all__ = [
    'CHUNK_FRAMES',
    'DEFAULT_FORMAT',
    'FormatConverter',
    'BufferWriteCallback',
    'BufferReadCallback',
    'StreamBuffer',
    'InputStreamHandler',
    'OutputStreamHandler',
    'BusListenerCallback',
    'BusListenerHandle',
    'AudioService'
]


CHUNK_FRAMES = 1764  # 40ms @ 44100Hz
"""Default chunk size in number of audio frames (samples per channel) used for buffering and streaming"""

DEFAULT_FORMAT: PCMFormat = PCMFormat(rate=44100, sample_fmt=PCMSampleFormat.int16, channels=1)
"""Default PCM audio format used by the audio service"""


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
PyAudioStreamCallback = Callable[[Optional[bytes], int, PyAudioStreamTimeInfo, int],
                                 PyAudioStreamCallbackReturn]
BufferReadCallback = Callable[[int], bytes]
BufferWriteCallback = Callable[[bytes, PCMFormat], None]


class FormatConverter:
    """
    Class used to convert a raw PCM audio stream from one format to another.
    Use one instance only for one continuous audio stream.
    """
    def __init__(self, source_format: PCMFormat, dest_format: PCMFormat):
        """
        Constructor for `FormatConverter`. Use only for one stream at a time.
        :param source_format: a `PCMFormat` object specifying the source PCM format
        :param dest_format: a `PCMFormat` object specifying the destination PCM format
        """
        if any(f.sample_fmt == PCMSampleFormat.float32 or f.channels not in (1, 2) for f in (source_format, dest_format)):
            raise ValueError('Only integer stereo or mono PCM samples are supported')
        self.source_format: PCMFormat = source_format
        self.dest_format: PCMFormat = dest_format
        self._ratecv_state = None  # Unknown type

    def convert(self, fragment: bytes) -> bytes:
        """
        Convert an audio fragment
        :param fragment: the source audio fragment in bytes
        :return: the converted audio fragment in bytes
        :raise ValueError: if the length of the source fragment is not a multiple of source_format.width
        """
        if len(fragment) % self.source_format.width != 0:
            raise ValueError('Length of audio fragment must be a multiple of "source_format.width"')
        if self.source_format.sample_fmt.width != self.dest_format.sample_fmt.width:
            fragment = audioop.lin2lin(fragment, self.source_format.sample_fmt.width, self.dest_format.sample_fmt.width)
        if self.source_format.rate != self.dest_format.rate:
            fragment, self._ratecv_state = audioop.ratecv(fragment, self.dest_format.sample_fmt.width,
                                                          self.source_format.channels,
                                                          self.source_format.rate, self.dest_format.rate,
                                                          self._ratecv_state)
        if self.source_format.channels != self.dest_format.channels:
            if self.source_format.channels == 2:
                assert self.dest_format.channels == 1
                fragment = audioop.tomono(fragment, self.dest_format.sample_fmt.width, 1.0, 1.0)
            elif self.source_format.channels == 1:
                assert self.dest_format.channels == 2
                fragment = audioop.tostereo(fragment, self.dest_format.sample_fmt.width, 1.0, 1.0)
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
        """Start time offset in seconds of the audio chunk relative to the beginning of the stream"""
        return self.start_offset / self.pcm_format.rate

    @property
    def start_time(self) -> float:
        """Absoulte start time of the audio chunk (usually from a monotonic clock)"""
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

    Objects of this class handle audio streams buffers from/to the audio backend (PortAudio)
    and provide an interface for control of the stream.
    """
    def __init__(self, audio_service: AudioService, bus: str, pcm_format: Optional[PCMFormat] = None):
        """
        Constructor for `StreamHandler`
        :param audio_service: a running `AudioService` instance
        :param bus: the audio bus name correlated with this stream
        :param pcm_format: the `PCMFormat` for the stream.
                           If not specified here, it must be provided before starting the acutual audio stream.
        """
        self._audio_service: AudioService = audio_service
        self._bus: str = bus
        self.pcm_format: Optional[PCMFormat] = pcm_format

        self._done_event: Event = Event()
        self._done_event.clear()
        self._stop_event: Event = Event()
        self._stop_event.clear()
        self.stream_error: Optional[BaseException] = None
        self._start_time: Optional[TimestampType] = None
        self._done_frames: int = 0

    @property
    @abstractmethod
    def direction(self) -> StreamDirection:
        """The direction of the stream"""

    @abstractmethod
    def _retrieve_audio_data(self, in_data: Optional[bytes], frame_count: int,
                             time_info: PyAudioStreamTimeInfo, status: int) -> bytes:
        """
        Retrieve the audio data chunk (input or output) for/from the backend
        :param in_data: input audio data in bytes. None if the stream is not an input stream.
        :param frame_count: number of frames (samples per channel) requested by PortAudio for output
        :param time_info: PortAudio's StreamTimeInfo for the current chunk
        :param status: PortAudio's status flags for the stream
        :return: the audio data in bytes for output (if an output stream) or the input audio data (same as in_data)
        """

    # noinspection PyUnusedLocal
    def pa_callback(self, in_data: Optional[bytes], frame_count: int,
                    time_info: PyAudioStreamTimeInfo, status: int) -> PyAudioStreamCallbackReturn:
        """
        Wrapper function used as callback for the PortAudio's stream.
        :param in_data: input audio data in bytes. None if the stream is not an input stream.
        :param frame_count: number of frames (samples per channel) requested by PortAudio for output
        :param time_info: PortAudio's StreamTimeInfo for the current chunk
        :param status: PortAudio's status flags for the stream
        :return: the audio data in bytes for output (if an output stream). None if the stream is not an output stream.
        """
        if self.pcm_format is None:
            self._audio_service.logger.error(f'Tried starting audio stream, '
                                             f'but pcm_format was never specified for {self.__class__.__name__}')
            return None, pyaudio.paAbort
        if self._start_time is None:
            self._start_time = time.monotonic()
        if self._stop_event.is_set():
            return None, pyaudio.paAbort
        try:
            audio_data: bytes = self._retrieve_audio_data(in_data=in_data, frame_count=frame_count,
                                                          time_info=time_info, status=status)
        except BaseException as exc:
            self._audio_service.logger.error(f'Error in {self.__class__.__name__}: {exc}', exc_info=True)
            return None, pyaudio.paAbort
        if not audio_data:
            self._done_event.set()
            return None, pyaudio.paComplete
        stream_buffer: StreamBuffer = StreamBuffer(audio_data, start_offset=self._done_frames, stream_handler=self)
        if not stream_buffer.check_size():
            self._audio_service.logger.debug(f'Audio buffer has bad size '
                                             f'(expected {len(audio_data)} % {self.pcm_format.width})')
        self._audio_service.route_buffer(stream_buffer)
        self._done_frames += len(audio_data) / self.pcm_format.width
        if self.direction == StreamDirection.INPUT:
            return None, pyaudio.paContinue
        elif self.direction == StreamDirection.OUTPUT:
            # If PyAudio gets less frames than requested, it stops the stream. Setting events just in case
            assume_complete: bool = stream_buffer.frames < frame_count
            pa_flag: int = pyaudio.paComplete if assume_complete else pyaudio.paContinue
            if assume_complete:
                self._done_event.set()
            return audio_data, pa_flag

    @property
    def bus(self) -> str:
        """The audio bus name from `AudioService` correlated with this stream"""
        return self._bus

    @property
    def success(self) -> bool:
        """True if audio stream is finished and no errors were caught, else False"""
        return self._done_event.is_set() and self.stream_error is None

    @property
    def done_event(self) -> Event:
        """Stream Event indicating whether the stream is done"""
        return self._done_event

    @property
    def stop_event(self) -> Event:
        """Stream Event indicating whether to stop the stream"""
        return self._stop_event

    @property
    def start_time(self) -> TimestampType:
        """Internal stream start time (usually a monotonic clock reference)"""
        if self._start_time is None:
            raise ValueError('Stream has not started yet')
        return self._start_time

    @property
    def done_frames(self) -> int:
        """Total number of audio frames streamed (samples per channel)"""
        return self._done_frames


class InputStreamHandler(StreamHandler):
    """
    Stream handler subclass for input streams
    """
    def __init__(self, *args, write_callback: Optional[BufferWriteCallback] = None, bus: Optional[str] = None,
                 pcm_format: Optional[PCMFormat] = None, **kwargs):
        """
        Constructor for `InputStreamHandler`

        :param args: additionl positional arguments for base `StreamHandler` constructor
        :param write_callback: an optional callable that is called with a new chunk of audio data when available
        :param bus: the audio bus name correlated with this stream. If omitted, defaults to `AudioService.BUS_INPUT`
        :param pcm_format: the `PCMFormat` requested for the input stream.
                           If not specified here, it defaults to the default PCM format.
        :param kwargs: additional keyword arguments for base `StreamHandler` constructor
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

    def _retrieve_audio_data(self, in_data: Optional[bytes], frame_count: int,
                             time_info: PyAudioStreamTimeInfo, status: int) -> bytes:
        audio_data: bytes = in_data
        if self.write_callback is not None and callable(self.write_callback):
            self.write_callback(audio_data, self.pcm_format)
        return audio_data


class OutputStreamHandler(StreamHandler):
    """
    Stream handler subclass for output streams
    """
    def __init__(self, *args, read_callback: Optional[BufferReadCallback] = None, bus: Optional[str] = None, **kwargs):
        """
        Constructor for `OutputStreamHandler`

        :param args: additionl positional arguments for base `StreamHandler` constructor
        :param read_callback: a callable that is called whenever a new chunk of audio data is requested
                              for output by the PortAudio backend.
                              Must accept one argument as the number of frames requested in integer
                              and must return the audio data chunk to output in bytes.
                              If not specified here, it must be provided before starting the actual audio stream.
        :param bus: the audio bus name correlated with this stream. If omitted, defaults to `AudioService.BUS_INPUT`
        :param kwargs: additional keyword arguments for base `StreamHandler` constructor
        """
        if bus is None:
            bus = AudioService.BUS_OUTPUT
        super().__init__(*args, bus=bus, **kwargs)
        self.read_callback: Optional[BufferReadCallback] = read_callback

    @property
    def direction(self) -> StreamDirection:
        return StreamDirection.OUTPUT

    def _retrieve_audio_data(self, in_data: Optional[bytes], frame_count: int,
                             time_info: PyAudioStreamTimeInfo, status: int) -> bytes:
        if self.read_callback is None:
            raise RuntimeError('Must provide a "read_callback" function in order to output audio')
        audio_data: bytes = self.read_callback(frame_count)
        return audio_data


BusListenerCallback = Callable[[StreamBuffer], Union[Awaitable[None], None]]


@dataclass(frozen=True)
class BusListenerHandle:
    """
    Dataclass used to represent a handle for a listener attached to an audio bus
    """
    bus_name: str
    """The bus name on which the listener is attached to"""

    callback: BusListenerCallback
    """The listener callback"""

    uuid: str = dataclass_field(default_factory=lambda: uuid.uuid1().hex, compare=True, hash=True)
    """An unique identifier for the handle. Mostly used for identification, comparison and hashing"""


class PlaybackCallable(Protocol):
    """Signature protocol specification for `AudioService` internal stream playback methods"""
    def __call__(self, *args, stream_handler: OutputStreamHandler, **kwargs) -> Awaitable: ...


@custom_log(component='AUDIO')
class AudioService(BackgroundService):
    """
    Class for the main audio service.
    Provides an unified interface to the audio backend and related audio functionality.
    """
    CHUNK_FRAMES: int = CHUNK_FRAMES
    """Default chunk size in number of audio frames (samples per channel) used for buffering and streaming"""
    DEFAULT_FORMAT: PCMFormat = DEFAULT_FORMAT
    """Default PCM audio format used by the audio service"""

    BUS_OUTPUT: str = 'output'
    """Default name for the output audio bus"""
    BUS_INPUT: str = 'input'
    """Default name for the input audio bus"""

    def __init__(self, input_device_index: Optional[int] = None, output_device_index: Optional[int] = None):
        """
        Constructor for `AudioService`
        :param input_device_index: The PortAudio input device index to use for input streams.
                                   If omitted, the default system's input device is used.
        :param output_device_index: The PortAudio output device index to use for output streams.
                                    If omitted, the default system's output device is used.
        """
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        super().__init__()
        self.input_device_index: Optional[int] = input_device_index
        self.output_device_index: Optional[int] = output_device_index
        self._pa: pyaudio.PyAudio = pyaudio.PyAudio()
        self._bus_listeners: List[BusListenerHandle] = []

    def _create_thread(self) -> Thread:
        return Thread(name='AudioThread', target=self._audio_thread_main)

    def _audio_thread_main(self) -> None:
        """Main entry point function for the audio thread"""
        try:
            self._loop.run_until_complete(self._audio_main_async())
        except SystemExit:
            pass
        finally:
            self.__log.debug('Audio thread ended')

    async def _audio_main_async(self) -> None:
        """Main entry point coroutine for the audio service (async)"""
        while True:
            if self._stop_event.is_set():
                raise SystemExit
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
            raise RuntimeError('Audio thread not running')

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        """Internal asyncio event loop related to the audio service functionality"""
        return self._loop

    def register_bus_listener(self, bus_name: str, callback: BusListenerCallback) -> BusListenerHandle:
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
            self.__log.warning(f'Attempted to remove unregistered bus listener: {repr(listener_handle)}')

    @contextmanager
    def bus_listener(self, bus_name: str, callback: BusListenerCallback) -> Iterator[None]:
        """
        Context manager utility method to register and unregister a bus listener within a with-block
        :param bus_name: the audio bus name on which to register the listener
        :param callback: the callback function for the listener
        """
        listener_handle: BusListenerHandle = self.register_bus_listener(bus_name, callback)
        try:
            yield
        finally:
            self.unregister_bus_listener(listener_handle)

    def route_buffer(self, stream_buffer: StreamBuffer) -> None:
        """
        Route a `StreamBuffer` object to all listeners of its `StreamHandler`'s related audio bus
        :param stream_buffer: the `StreamBuffer` object to route
        """
        for listener_handle in self._bus_listeners:
            if listener_handle.bus_name == stream_buffer.stream_handler.bus:
                self._loop.create_task(self._route_asap(listener_handle, stream_buffer))

    @staticmethod
    async def _route_asap(listener_handle: BusListenerHandle, stream_buffer: StreamBuffer) -> None:
        """
        Actual internal async coroutine that routes a `StreamBuffer` to a bus listener
        :param listener_handle: the listener handle to which to route the buffer to
        :param stream_buffer: the audio stream buffer object
        """
        if iscoroutinefunction(listener_handle.callback):
            await listener_handle.callback(stream_buffer)
        else:
            listener_handle.callback(stream_buffer)

    async def _pa_stream(self, stream_handler: StreamHandler, direction: StreamDirection):
        """
        Handle a PortAudio stream
        :param stream_handler: the `StreamHandler` object to associate to the stream
        :param direction: the direction of the audio stream
        """
        direction_kwargs: MutableMapping[str, Any]
        if direction == StreamDirection.INPUT:
            direction_kwargs = dict(input=True, input_device_index=self.input_device_index)
        elif direction == StreamDirection.OUTPUT:
            direction_kwargs = dict(output=True, output_device_index=self.output_device_index)
        else:
            raise AttributeError(f'Invalid direction for PyAudio stream: {direction}')
        with closing(self._pa.open(**stream_handler.pcm_format.pyaudio_args, frames_per_buffer=self.CHUNK_FRAMES,
                                   **direction_kwargs, stream_callback=stream_handler.pa_callback)) as stream:
            try:
                stream.start_stream()
                try:
                    while stream.is_active() and not stream_handler.stop_event.is_set():
                        await asyncio.sleep(0.02)
                finally:
                    stream.stop_stream()
            except SystemExit:
                pass
            except BaseException as exc:
                self.__log.warning(f'Error in PyAudio stream: {exc}', exc_info=True)
                stream_handler.stream_error = exc
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
    def audio_input(self, write_callback: Optional[BufferWriteCallback] = None,
                    pcm_format: Optional[PCMFormat] = None) -> Iterator[InputStreamHandler]:
        """
        Context manager utility method to start and stop an audio input stream
        :param write_callback: an optional callable that is called with a new chunk of audio data when available
        :param pcm_format: the `PCMFormat` requested for the input stream.
                           If not specified here, it defaults to the default PCM format.
        :return: yields the `InputStreamHandler` for the input stream
        """
        self.ensure_running()
        stream_handler: InputStreamHandler = InputStreamHandler(audio_service=self,
                                                                write_callback=write_callback,
                                                                pcm_format=pcm_format)
        self._loop.create_task(self._pa_acquire(stream_handler=stream_handler))
        try:
            yield stream_handler
        except SystemExit:
            pass
        except BaseException as exc:
            self.__log.warning(f'Error in input PyAudio stream: {exc}', exc_info=True)
            stream_handler.stream_error = exc
            raise
        finally:
            stream_handler.stop_event.set()

    async def _pa_playback(self, stream_handler: OutputStreamHandler) -> None:
        """
        Handle an output PortAudio stream
        :param stream_handler: the `OutputStreamHandler` object to associate to the stream
        """
        await self._pa_stream(stream_handler, StreamDirection.OUTPUT)

    async def _play_wav(self, filepath: str, stream_handler: OutputStreamHandler) -> None:
        """
        Coroutine to play a wave file using the `wave` library
        :param filepath: the path of the .wav file to play
        :param stream_handler: an instantiated `OutputStreamHandler` to associate to the output stream
        """
        with wave.open(filepath, 'rb') as wavefile:
            pcm_format: PCMFormat = PCMFormat(rate=wavefile.getframerate(),
                                              sample_fmt=PCMSampleFormat.get_format_from_width(wavefile.getsampwidth()),
                                              channels=wavefile.getnchannels())
            stream_handler.read_callback = wavefile.readframes
            stream_handler.pcm_format = pcm_format
            await self._pa_playback(stream_handler)

    @asynccontextmanager
    async def _ffmpeg_decoder(self, input_args: MutableMapping[str, Any], pcm_format: PCMFormat,
                              stream_handler: OutputStreamHandler, pipe_stdin: bool) -> Iterator[subprocess.Popen]:
        """
        Async context manager utility method to wrap a FFmpeg-decoded playback stream
        :param input_args: commandline args for FFmpeg for the input audio to be decoded
        :param pcm_format: the requested PCM output format for the stream
        :param stream_handler: an instantiated `OutputStreamHandler` to associate to the output stream
        :param pipe_stdin: True if the input audio is expected to be read from FFmpeg's stdin, else False
        :return: yields the FFmpeg subprocess
        """
        ffmpeg_spec: ffmpeg.Stream = ffmpeg.input(**input_args).output('pipe:', **pcm_format.ffmpeg_args)
        ffmpeg_args: List[str] = ffmpeg.compile(ffmpeg_spec, 'ffmpeg')
        ffmpeg_process: subprocess.Popen = subprocess.Popen(args=ffmpeg_args, bufsize=0, text=False,
                                                            stdin=subprocess.PIPE if pipe_stdin else subprocess.DEVNULL,
                                                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                                            close_fds=True)

        def read_stdout_pipe(frame_count: int) -> bytes:
            nonlocal ffmpeg_process, pcm_format
            # Ensuring we don't read less than what requested, until output has finished
            target_bytes: int = frame_count * pcm_format.width
            buffer: bytes = b''
            while (bytes_to_read := target_bytes - len(buffer)) > 0:
                read_bytes: bytes = ffmpeg_process.stdout.read(bytes_to_read)
                if not read_bytes:
                    if not pipe_stdin or ffmpeg_process.stdin.closed:
                        if ffmpeg_process.stdout and not ffmpeg_process.stdout.closed:
                            ffmpeg_process.stdout.close()
                        break
                    time.sleep(0.5 * (bytes_to_read / pcm_format.width) * pcm_format.sample_duration)
                buffer += read_bytes
            return buffer

        stream_handler.read_callback = read_stdout_pipe
        stream_handler.pcm_format = pcm_format
        playback_task: asyncio.Task = self._loop.create_task(self._pa_playback(stream_handler))

        try:
            yield ffmpeg_process
        except SystemExit:
            pass
        except BrokenPipeError as exc:
            self.__log.warning(f'Stopped playback: {exc}')
        except BaseException as exc:
            self.__log.warning(f'Got exception in FFmpeg decoder context: {exc}')
            stream_handler.stream_error = exc
            raise
        finally:
            ffmpeg_retcode: Optional[int]
            while True:
                ffmpeg_retcode = ffmpeg_process.poll()
                if playback_task.done() or ffmpeg_retcode is not None:
                    stream_handler.stop_event.set()
                    break
                await asyncio.sleep(0.1)
            if ffmpeg_retcode is None:
                for _ in range(20):
                    ffmpeg_retcode = ffmpeg_process.poll()
                    if ffmpeg_retcode is not None:
                        break
                    await asyncio.sleep(0.1)
                else:
                    error_msg: str = 'FFmpeg still running 2.0s after output stream ended, killing it'
                    self.__log.debug(error_msg)
                    ffmpeg_process.kill()
                    stream_handler.stream_error = RuntimeError(error_msg)
                    return
            if ffmpeg_retcode != 0:  # TODO: Retrieve stderr?
                error_msg: str = f'FFmpeg exited with return code {ffmpeg_retcode}'
                self.__log.debug(error_msg)
                stream_handler.stream_error = RuntimeError(error_msg)

    async def _play_ffmpeg_piped(self, audio: Union[bytes, IO], stream_handler: OutputStreamHandler,
                                 codec: Optional[str] = None, data_pcm_format: Optional[PCMFormat] = None,
                                 pcm_format: Optional[PCMFormat] = None) -> None:
        """
        Coroutine to play a FFmpeg-decoded stream piped to stdin from a buffer or bytes string
        :param audio: the audio buffer (stream) object or bytes string
        :param stream_handler: an instantiated `OutputStreamHandler` to associate to the output stream
        :param codec: the codec used to decode the input audio data, optional
        :param data_pcm_format: the PCM format for the input audio data, optional
        :param pcm_format: the requested PCM output format for the stream, if omitted the default PCM format will be used
        """
        if pcm_format is None:
            pcm_format = self.DEFAULT_FORMAT
        input_args: MutableMapping[str, Any] = dict(filename='pipe:')
        if codec is not None:
            input_args['acodec'] = codec
        if data_pcm_format is not None:
            input_args.update(**data_pcm_format.ffmpeg_args)
        async with self._ffmpeg_decoder(input_args, pcm_format, stream_handler, pipe_stdin=True) as ffmpeg_process:
            chunk_size: int
            if data_pcm_format is not None:
                chunk_size = ((self.CHUNK_FRAMES * data_pcm_format.rate) // pcm_format.rate) * data_pcm_format.width
            else:
                chunk_size = self.CHUNK_FRAMES
            # Feed audio data to ffmpeg stdin
            is_stream: bool = hasattr(audio, 'read')
            if not is_stream:
                assert isinstance(audio, bytes)
                chunks_generator: Iterator[bytes] = chunked(audio, chunk_size)
            while True:
                input_chunk: bytes
                if is_stream:
                    input_chunk = audio.read(chunk_size)  # FIXME: Careful with blocking reads within async
                    if not input_chunk:
                        audio.close()
                        break
                else:
                    try:
                        input_chunk = next(chunks_generator)
                    except StopIteration:
                        break
                if stream_handler.stop_event.is_set():
                    break
                if not ffmpeg_process.stdin or ffmpeg_process.stdin.closed or ffmpeg_process.poll() is not None:
                    raise BrokenPipeError('FFmpeg subprocess stdin is closed')
                try:
                    ffmpeg_process.stdin.write(input_chunk)
                except OSError as exc:
                    if 'Errno 22' in str(exc):
                        raise BrokenPipeError(str(exc)) from OSError
                    raise
                await asyncio.sleep(0)
            ffmpeg_process.stdin.close()

    async def _play_ffmpeg_file(self, filepath: str,
                                stream_handler: OutputStreamHandler, pcm_format: Optional[PCMFormat] = None) -> None:
        """
        Coroutine to play a FFmpeg-decoded file
        :param filepath: the path of the audio file
        :param stream_handler: an instantiated `OutputStreamHandler` to associate to the output stream
        :param pcm_format: the requested PCM output format for the stream, if omitted the default PCM format will be used
        """
        if pcm_format is None:
            pcm_format = self.DEFAULT_FORMAT
        input_args: MutableMapping[str, Any] = dict(filename=filepath)
        async with self._ffmpeg_decoder(input_args, pcm_format, stream_handler, pipe_stdin=False):
            pass

    def _play_stream(self, playback_callable: PlaybackCallable, blocking: bool) -> OutputStreamHandler:
        """
        Internal wrapper method to start a playback stream
        :param playback_callable: a callable that when called passing a `OutputStreamHandler`
                                  will return the coroutine to schedule to the asyncio loop for playback
        :param blocking: if True, will wait until the stream to end to return
        :return: the `OutputStreamHandler` associated with the playback stream
        """
        self.ensure_running()
        stream_handler: OutputStreamHandler = OutputStreamHandler(audio_service=self)
        self._loop.create_task(playback_callable(stream_handler=stream_handler))
        if blocking:
            stream_handler.done_event.wait()
        return stream_handler

    def play_data(self, audio: Union[bytes, IO],
                  codec: Optional[str] = None, data_pcm_format: Optional[PCMFormat] = None,
                  blocking: bool = True) -> OutputStreamHandler:
        """
        Start a playback stream for audio data in a binary stream or bytes string
        :param audio: the audio buffer (stream) object or bytes string
        :param codec: the codec used to decode the input audio data, optional
        :param data_pcm_format: the PCM format for the input audio data, optional
        :param blocking: if True, will wait until the stream to end to return
        :return: the `OutputStreamHandler` associated with the playback stream
        """
        # noinspection PyTypeChecker
        playback_callable: PlaybackCallable = partial(self._play_ffmpeg_piped, audio=audio,
                                                      codec=codec, data_pcm_format=data_pcm_format)
        return self._play_stream(playback_callable, blocking=blocking)

    def play_file(self, filepath: str, blocking: bool = True) -> OutputStreamHandler:
        """
        Start a playback stream for an audio file
        :param filepath: the path of the audio file
        :param blocking: if True, will wait until the stream to end to return
        :return: the `OutputStreamHandler` associated with the playback stream
        """
        # noinspection PyTypeChecker
        playback_callable: PlaybackCallable = partial(self._play_ffmpeg_file, filepath=filepath)
        return self._play_stream(playback_callable, blocking=blocking)

    def play_descriptor(self, audio_descriptor: AudioDescriptor, blocking: bool = True) -> OutputStreamHandler:
        """
        Start a playback stream for the giving audio descriptor, automatically picking the most appropriate method
        :param audio_descriptor: the audio descriptor that specifies playback
        :param blocking: if True, will wait until the stream to end to return
        :return: the `OutputStreamHandler` associated with the playback stream
        """
        try:
            if isinstance(audio_descriptor, AudioFileDescriptor):
                return self.play_file(audio_descriptor.path, blocking=blocking)
            elif isinstance(audio_descriptor, (AudioBytesDescriptor, AudioStreamDescriptor)):
                audio: Union[bytes, IO]
                if isinstance(audio_descriptor, AudioBytesDescriptor):
                    audio = audio_descriptor.audio_data
                elif isinstance(audio_descriptor, AudioStreamDescriptor):
                    audio = audio_descriptor.audio_stream
                else:
                    raise SyntaxError(f'wrong class, sanity check failed')
                if isinstance(audio_descriptor, AudioPCMDescriptor):
                    return self.play_data(audio,
                                          data_pcm_format=audio_descriptor.pcm_format,
                                          blocking=blocking)
                elif isinstance(audio_descriptor, AudioEncodedDescriptor):
                    return self.play_data(audio,
                                          codec=audio_descriptor.codec,
                                          blocking=blocking)
                else:
                    raise ValueError(f'unknown or unhandled audio descriptor type')
            else:
                raise ValueError(f'unknown or unhandled audio descriptor type')
        except ValueError as exc:
            raise ValueError(f'Could not play audio descriptor {repr(audio_descriptor)}: {exc}') from exc
