from __future__ import annotations

import asyncio
import audioop
import enum
import subprocess
import time
import uuid
import wave
from abc import ABC, abstractmethod
from collections import namedtuple
from contextlib import closing, contextmanager, asynccontextmanager
from dataclasses import dataclass, field as dataclass_field
from functools import partial
from inspect import iscoroutinefunction
from threading import Event, Thread
from typing import Callable, TypedDict, Optional, Tuple, List, MutableMapping, Any, Union, Awaitable, Protocol

import ffmpeg
import pyaudio
import numpy as np

from ..common import BackgroundService, TimestampType, chunked
from ..logger import custom_log


__all__ = [
    'CHUNK_FRAMES',
    'PCMSampleFormat',
    'PCMFormat',
    'DEFAULT_FORMAT',
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


DEFAULT_FORMAT: PCMFormat = PCMFormat(rate=44100, sample_fmt=PCMSampleFormat.int16, channels=1)


class PyAudioStreamTimeInfo(TypedDict):
    input_buffer_adc_time: float
    current_time: float
    output_buffer_dac_time: float


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
        if any(f.sample_fmt == PCMSampleFormat.float32 or f.channels not in (1, 2) for f in (source_format, dest_format)):
            raise ValueError('Only integer stereo or mono PCM samples are supported')
        self.source_format: PCMFormat = source_format
        self.dest_format: PCMFormat = dest_format
        self._ratecv_state = None

    def convert(self, fragment: bytes) -> bytes:
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
    buffer_data: bytes
    start_offset: int
    stream_handler: StreamHandler

    @property
    def size(self) -> int:
        return len(self.buffer_data)

    @property
    def pcm_format(self) -> PCMFormat:
        return self.stream_handler.pcm_format

    def check_size(self) -> bool:
        return self.size % self.pcm_format.width == 0

    @property
    def frames(self) -> int:
        return self.size // self.pcm_format.width

    @property
    def duration(self) -> float:
        return self.frames / self.pcm_format.rate

    @property
    def start_time_relative(self) -> float:
        return self.start_offset / self.pcm_format.rate

    @property
    def start_time(self) -> float:
        return self.stream_handler.start_time + self.start_time_relative

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration


class StreamDirection(enum.Enum):
    INPUT = enum.auto()
    OUTPUT = enum.auto()


class StreamHandler(ABC):
    DEFAULT_BUS: Optional[str] = None

    def __init__(self, audio_service: AudioService, bus: str, pcm_format: Optional[PCMFormat] = None):
        self._audio_service: AudioService = audio_service
        self._bus: str = bus
        self.pcm_format: Optional[PCMFormat] = pcm_format

        self._done_event: Event = Event()
        self._done_event.clear()
        self._stop_event: Event = Event()
        self._stop_event.clear()
        self._start_time: Optional[TimestampType] = None
        self._done_frames: int = 0

    @property
    @abstractmethod
    def direction(self) -> StreamDirection:
        ...

    @abstractmethod
    def _retrieve_audio_data(self, in_data: Optional[bytes], frame_count: int,
                             time_info: PyAudioStreamTimeInfo, status: int) -> bytes:
        ...

    # noinspection PyUnusedLocal
    def pa_callback(self, in_data: Optional[bytes], frame_count: int,
                    time_info: PyAudioStreamTimeInfo, status: int) -> PyAudioStreamCallbackReturn:
        if self._start_time is None:
            self._start_time = time.monotonic()
        # print(time_info)
        if self._stop_event.is_set():
            return None, pyaudio.paAbort
        audio_data: bytes = self._retrieve_audio_data(in_data=in_data, frame_count=frame_count,
                                                      time_info=time_info, status=status)
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
        return self._bus

    @property
    def done_event(self) -> Event:
        return self._done_event

    @property
    def stop_event(self) -> Event:
        return self._stop_event

    @property
    def start_time(self) -> TimestampType:
        if self._start_time is None:
            raise ValueError('Stream has not started yet')
        return self._start_time

    @property
    def done_frames(self) -> int:
        return self._done_frames


class InputStreamHandler(StreamHandler):
    def __init__(self, *args, write_callback: Optional[BufferWriteCallback] = None, bus: Optional[str] = None,
                 pcm_format: Optional[PCMFormat] = None, **kwargs):
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
    def __init__(self, *args, read_callback: Optional[BufferReadCallback] = None, bus: Optional[str] = None, **kwargs):
        if bus is None:
            bus = AudioService.BUS_OUTPUT
        super().__init__(*args, bus=bus, **kwargs)
        self.read_callback: Optional[BufferReadCallback] = read_callback

    @property
    def direction(self) -> StreamDirection:
        return StreamDirection.OUTPUT

    def _retrieve_audio_data(self, in_data: Optional[bytes], frame_count: int,
                             time_info: PyAudioStreamTimeInfo, status: int) -> bytes:
        assert self.read_callback is not None
        if self.read_callback is None:
            raise RuntimeError('Must provide a "read_callback" function in order to output audio')
        audio_data: bytes = self.read_callback(frame_count)
        return audio_data


BusListenerCallback = Callable[[StreamBuffer], Union[Awaitable[None], None]]


@dataclass(frozen=True)
class BusListenerHandle:
    bus_name: str
    callback: BusListenerCallback
    uuid: str = dataclass_field(default_factory=lambda: uuid.uuid1().hex, compare=True, hash=True)


class PlaybackCallable(Protocol):
    def __call__(self, *args, stream_handler: OutputStreamHandler, **kwargs) -> Awaitable: ...


@custom_log(component='AUDIO')
class AudioService(BackgroundService):
    CHUNK_FRAMES: int = CHUNK_FRAMES
    DEFAULT_FORMAT: PCMFormat = DEFAULT_FORMAT

    BUS_OUTPUT: str = 'output'
    BUS_INPUT: str = 'input'

    def __init__(self, input_device_index: Optional[int] = None, output_device_index: Optional[int] = None):
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        super().__init__()
        self.input_device_index: Optional[int] = input_device_index
        self.output_device_index: Optional[int] = output_device_index
        self._pa = pyaudio.PyAudio()
        self._bus_listeners: List[BusListenerHandle] = []

    def _create_thread(self) -> Thread:
        return Thread(name='AudioThread', target=self._audio_thread_main)

    def _audio_thread_main(self) -> None:
        try:
            self._loop.run_until_complete(self._audio_main_async())
        except SystemExit:
            pass
        finally:
            self.__log.debug('Audio thread ended')

    async def _audio_main_async(self) -> None:
        while True:
            if self._stop_event.is_set():
                raise SystemExit
            await asyncio.sleep(0.1)

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    def ensure_running(self):
        if not self.running:
            raise RuntimeError('Audio thread not running')

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        return self._loop

    def register_bus_listener(self, bus_name: str, callback: BusListenerCallback) -> BusListenerHandle:
        listener_handle: BusListenerHandle = BusListenerHandle(bus_name, callback)
        self._bus_listeners.append(listener_handle)
        return listener_handle

    def unregister_bus_listener(self, listener_handle: BusListenerHandle) -> None:
        if listener_handle in self._bus_listeners:
            self._bus_listeners.remove(listener_handle)
        else:
            self.__log.warning(f'Attempted to remove unregistered bus listener: {repr(listener_handle)}')

    @contextmanager
    def bus_listener(self, bus_name: str, callback: BusListenerCallback):
        listener_handle: BusListenerHandle = self.register_bus_listener(bus_name, callback)
        try:
            yield
        finally:
            self.unregister_bus_listener(listener_handle)

    def route_buffer(self, stream_buffer: StreamBuffer):
        for listener_handle in self._bus_listeners:
            if listener_handle.bus_name == stream_buffer.stream_handler.bus:
                self._loop.create_task(self._route_asap(listener_handle, stream_buffer))

    @staticmethod
    async def _route_asap(listener_handle: BusListenerHandle, stream_buffer: StreamBuffer):
        if iscoroutinefunction(listener_handle.callback):
            await listener_handle.callback(stream_buffer)
        else:
            listener_handle.callback(stream_buffer)

    async def _pa_stream(self, stream_handler: StreamHandler, direction: StreamDirection) -> None:
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
            finally:
                stream_handler.done_event.set()
                stream_handler.stop_event.set()

    async def _pa_acquire(self, stream_handler: StreamHandler) -> None:
        await self._pa_stream(stream_handler, StreamDirection.INPUT)

    @contextmanager
    def audio_input(self, write_callback: Optional[BufferWriteCallback] = None,
                    pcm_format: Optional[PCMFormat] = None) -> InputStreamHandler:
        self.ensure_running()
        stream_handler: InputStreamHandler = InputStreamHandler(audio_service=self,
                                                                write_callback=write_callback,
                                                                pcm_format=pcm_format)
        self._loop.create_task(self._pa_acquire(stream_handler=stream_handler))
        try:
            yield stream_handler
        finally:
            stream_handler.stop_event.set()

    async def _pa_playback(self, stream_handler: StreamHandler) -> None:
        await self._pa_stream(stream_handler, StreamDirection.OUTPUT)

    async def _play_wav(self, filepath: str, stream_handler: StreamHandler) -> None:
        with wave.open(filepath, 'rb') as wavefile:
            pcm_format: PCMFormat = PCMFormat(rate=wavefile.getframerate(),
                                              sample_fmt=PCMSampleFormat.get_format_from_width(wavefile.getsampwidth()),
                                              channels=wavefile.getnchannels())
            stream_handler.read_callback = wavefile.readframes
            stream_handler.pcm_format = pcm_format
            await self._pa_playback(stream_handler)

    @asynccontextmanager
    async def _ffmpeg_decoder(self, input_args: MutableMapping[str, Any], pcm_format: PCMFormat,
                              stream_handler: OutputStreamHandler, pipe_stdin: bool):
        ffmpeg_spec: ffmpeg.Stream = ffmpeg.input(**input_args).output('pipe:', **pcm_format.ffmpeg_args)
        ffmpeg_args: List[str] = ffmpeg.compile(ffmpeg_spec, 'ffmpeg')
        ffmpeg_process: subprocess.Popen = subprocess.Popen(args=ffmpeg_args, bufsize=0, text=False,
                                                            stdin=subprocess.PIPE if pipe_stdin else subprocess.DEVNULL,
                                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        def read_stdout_pipe(frame_count: int) -> bytes:
            nonlocal ffmpeg_process
            return ffmpeg_process.stdout.read(frame_count * pcm_format.width)

        stream_handler.read_callback = read_stdout_pipe
        stream_handler.pcm_format = pcm_format
        playback_task: asyncio.Task = self._loop.create_task(self._pa_playback(stream_handler))

        try:
            yield ffmpeg_process

        except BaseException as exc:
            self.__log.warning(f'Got exception in FFmpeg decoder context: {exc}')
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
                    self.__log.debug('FFmpeg still running 2.0s after output stream ended, killing it')
                    ffmpeg_process.kill()
                    return
            if ffmpeg_retcode != 0:  # TODO: Retrieve stderr?
                self.__log.debug(f'FFmpeg exited with return code {ffmpeg_retcode}')

    async def _play_ffmpeg_bytes(self, audio_data: bytes, data_pcm_format: PCMFormat,
                                 stream_handler: OutputStreamHandler, pcm_format: Optional[PCMFormat] = None) -> None:
        if pcm_format is None:
            pcm_format = self.DEFAULT_FORMAT
        input_args: MutableMapping[str, Any] = dict(filename='pipe:', **data_pcm_format.ffmpeg_args)
        async with self._ffmpeg_decoder(input_args, pcm_format, stream_handler, pipe_stdin=True) as ffmpeg_process:
            chunk_size: int = ((self.CHUNK_FRAMES * data_pcm_format.rate) // pcm_format.rate) * data_pcm_format.width
            # Feed audio data to ffmpeg stdin
            for input_chunk in chunked(audio_data, chunk_size):
                ffmpeg_process.stdin.write(input_chunk)
                await asyncio.sleep(0)
            ffmpeg_process.stdin.close()

    async def _play_ffmpeg_file(self, filepath: str,
                                stream_handler: OutputStreamHandler, pcm_format: Optional[PCMFormat] = None) -> None:
        if pcm_format is None:
            pcm_format = self.DEFAULT_FORMAT
        input_args: MutableMapping[str, Any] = dict(filename=filepath)
        async with self._ffmpeg_decoder(input_args, pcm_format, stream_handler, pipe_stdin=False):
            pass

    def _play_stream(self, playback_callable: PlaybackCallable, blocking: bool) -> Optional[OutputStreamHandler]:
        self.ensure_running()
        stream_handler: OutputStreamHandler = OutputStreamHandler(audio_service=self)
        self._loop.create_task(playback_callable(stream_handler=stream_handler))
        if not blocking:
            return stream_handler
        else:
            stream_handler.done_event.wait()

    def play_bytes(self, audio_data: bytes, data_pcm_format: PCMFormat,
                   blocking: bool = True) -> Optional[OutputStreamHandler]:
        # noinspection PyTypeChecker
        playback_callable: PlaybackCallable = partial(self._play_ffmpeg_bytes,
                                                      audio_data=audio_data, data_pcm_format=data_pcm_format)
        return self._play_stream(playback_callable, blocking=blocking)

    def play_file(self, filepath: str, blocking: bool = True) -> Optional[OutputStreamHandler]:
        # noinspection PyTypeChecker
        playback_callable: PlaybackCallable = partial(self._play_ffmpeg_file, filepath=filepath)
        return self._play_stream(playback_callable, blocking=blocking)
