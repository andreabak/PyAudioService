"""Audio recording functionality"""

import asyncio
import audioop
import os
import subprocess
import time
from asyncio.subprocess import Process
from collections import deque
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field as dataclass_field
from threading import Event
from typing import List, MutableMapping, Any, Optional, Union, Sequence, Deque, AsyncContextManager, ContextManager

import ffmpeg
import numpy as np

from ..logger import custom_log
from .service import StreamBuffer, DEFAULT_FORMAT, AudioService, CHUNK_FRAMES, write_to_async_pipe_sane
from .datatypes import PCMSampleFormat, PCMFormat


__all__ = [
    'AudioRecorder'
]


@dataclass
class StreamBuffersTimeFrame:
    """
    A dataclass where received `StreamBuffer`s for a certain time frame are stored
    """
    start_time: float = dataclass_field(default_factory=time.monotonic)
    buffers: List[StreamBuffer] = dataclass_field(default_factory=list)


@custom_log(component='RECORDING')
class AudioRecorder:  # TODO: Improve logging for class
    """
    Class used for audio recording to a file from one or more audio buses within `AudioService`
    """
    FRAMES_DELAY: int = 10
    """Time frames to buffer before combining audio chunks and saving them"""

    INTERNAL_FORMAT: PCMFormat = PCMFormat(rate=DEFAULT_FORMAT.rate,
                                           sample_fmt=PCMSampleFormat.float32,
                                           channels=DEFAULT_FORMAT.channels)
    """Internal PCM format used for audio data manipulations"""

    DEFAULT_ENCODER_OPTIONS: MutableMapping[str, Any] = {'c:a': 'aac', 'q:a': '192k'}
    """Default FFmpeg audio encoder commandline options"""

    def __init__(self, audio_service: AudioService, out_file_path: str, source_buses: Union[Sequence[str], str],
                 pcm_format: Optional[PCMFormat] = None,
                 encoder_options: Optional[MutableMapping[str, Any]] = None):
        """
        Constructor for `AudioRecorder`
        :param audio_service: the `AudioService` instance from which to record audio
        :param out_file_path: path for the recorded audio file
        :param source_buses: the audio buses within the `AudioService` instance from which to record audio.
                             Can be either a sequence of bus names, or a string for a single bus.
        :param pcm_format: the PCM format used for the output recorded audio file (sample format might be ignored).
                           If omitted, the default PCM format will be used.
        :param encoder_options: FFmpeg audio encoder commandline options. If omitted the default ones will be used.
        """
        self._audio_service: AudioService = audio_service
        self._out_file_path: str = out_file_path
        if not source_buses:
            raise ValueError('Must specify at least one source bus for recording')
        if isinstance(source_buses, str):
            source_buses = [source_buses]
        self._source_buses: Sequence[str] = source_buses
        if pcm_format is None:
            pcm_format = DEFAULT_FORMAT
        self._pcm_format: PCMFormat = pcm_format
        if encoder_options is None:
            encoder_options = self.DEFAULT_ENCODER_OPTIONS
        self._encoder_options: MutableMapping[str, Any] = encoder_options

        self._time_frames: Deque[StreamBuffersTimeFrame] = deque()
        self._frame_size: int = CHUNK_FRAMES
        self._ffmpeg_process: Optional[Process] = None
        self._recording: bool = False
        self._stop_event: Event = Event()

    @property
    def stop_event(self) -> Event:
        """Internal Event that signals recording to stop"""
        return self._stop_event

    def start(self) -> None:
        """Start the audio recording"""
        self.__log.info(f'Starting recording for {"+".join(self._source_buses)} '
                        f'on {os.path.basename(self._out_file_path)}')
        self._audio_service.ensure_running()
        asyncio.run_coroutine_threadsafe(self._start_recording(), self._audio_service.loop)

    def stop(self) -> None:
        """Stop the audio recording"""
        self._stop_event.set()

    @contextmanager
    def record(self) -> ContextManager:
        """Context manager utility method to start and stop recording within a with-block"""
        self.start()
        try:
            yield
        finally:
            self.stop()

    async def _start_recording(self) -> None:
        """Internal coroutine that starts the audio recording within `AudioService`'s asyncio loop"""
        ffmpeg_spec: ffmpeg.Stream = ffmpeg.input('pipe:', **self.INTERNAL_FORMAT.ffmpeg_args) \
                                           .filter('alimiter', attack=10, release=20) \
                                           .output(self._out_file_path,
                                                   **self._pcm_format.ffmpeg_args_nofmt, **self._encoder_options)
        ffmpeg_context: AsyncContextManager[Process] = self._audio_service.ffmpeg_subprocess(
            ffmpeg_spec, stdin=subprocess.PIPE, kill_timeout=5.0, logger=self.__log
        )
        async with ffmpeg_context as ffmpeg_process:
            self._ffmpeg_process = ffmpeg_process
            try:
                try:
                    await self._recording_loop()
                except SystemExit:
                    pass
                await self._time_frames_cleanup()
            except BrokenPipeError as exc:
                self.__log.info(f'Force stopped recording: {exc}')
                raise

    async def _recording_loop(self) -> None:
        """Internal coroutine that handles the recording loop within `AudioService`'s asyncio loop"""
        time_step: float = self._frame_size * self._pcm_format.sample_duration
        start_time: float = time.monotonic()
        last_tf_time: Optional[float] = None
        self._recording = True
        try:
            with ExitStack() as buses_stack:
                for bus in self._source_buses:  # Attach recorder to all buses
                    buses_stack.enter_context(self._audio_service.bus_listener(bus, self.record_buffer))

                while True:
                    current_tf_time: float
                    if last_tf_time is None:
                        current_tf_time = start_time
                    else:
                        current_tf_time = last_tf_time + time_step

                    time_frame: StreamBuffersTimeFrame = StreamBuffersTimeFrame(start_time=current_tf_time)
                    self._time_frames.append(time_frame)

                    if self._stop_event.is_set() or self._audio_service.check_exiting(dont_raise=True):
                        raise SystemExit

                    if len(self._time_frames) > self.FRAMES_DELAY:
                        await self._save_time_frame(self._time_frames.popleft())

                    sleep_delay: float = current_tf_time - time.monotonic() + time_step
                    if (time_shift := time_step - sleep_delay) > time_step:
                        self.__log.debug(f'Got some {time_shift*1000:.2f}ms of time shift on recording loop')
                    await asyncio.sleep(max(0.0, sleep_delay))
                    last_tf_time = current_tf_time
        finally:
            self._recording = False

    async def _time_frames_cleanup(self) -> None:
        """Internal coroutine that cleans up and saves pending time frames"""
        while self._time_frames:
            await self._save_time_frame(self._time_frames.popleft())
        write_to_async_pipe_sane(self._ffmpeg_process, self._ffmpeg_process.stdin, b'\0')
        self._ffmpeg_process.stdin.close()

    async def _save_time_frame(self, time_frame: StreamBuffersTimeFrame) -> None:
        """
        Internal coroutine that handles buffers merging and saving piping to FFmpeg for the given time frame
        :param time_frame: the `StreamBuffersTimeFrame` time frame object to save
        """
        internal_fmt_channels: int = self.INTERNAL_FORMAT.channels
        internal_fmt_numpy: np.number = self.INTERNAL_FORMAT.sample_fmt.numpy
        # Okay...
        cum_buffer: np.ndarray = np.zeros(self._frame_size * internal_fmt_channels, dtype=internal_fmt_numpy)
        for stream_buffer in time_frame.buffers:
            buffer_data: bytes = stream_buffer.buffer_data
            # TODO: Implement time offset padding compensation
            sample_format: PCMSampleFormat = stream_buffer.stream_handler.pcm_format.sample_fmt
            if sample_format == PCMSampleFormat.int24:  # 24-bit is not supported by numpy, convert to int32
                intermediate_format: PCMSampleFormat = PCMSampleFormat.int32
                buffer_data = audioop.lin2lin(buffer_data, sample_format.width, intermediate_format.width)
                sample_format = intermediate_format
            buffer_np: np.ndarray = np.frombuffer(buffer_data, dtype=sample_format.numpy).astype(internal_fmt_numpy)
            if self.INTERNAL_FORMAT.sample_fmt == PCMSampleFormat.float32 and sample_format != self.INTERNAL_FORMAT.sample_fmt:
                buffer_np /= (2 ** (8 * sample_format.width)) / 2  # Normalize to (-1.0, 1.0) if converting to float
            if (end_padding := cum_buffer.size - buffer_np.size) > 0:
                buffer_np = np.pad(buffer_np, (0, end_padding), mode='constant')  # TODO: offset padding compensation
            elif buffer_np.size > cum_buffer.size:
                raise NotImplementedError  # TODO: Implement
            cum_buffer += buffer_np
        out_buffer: bytes = cum_buffer.tobytes()

        write_to_async_pipe_sane(self._ffmpeg_process, self._ffmpeg_process.stdin, out_buffer)

    async def record_buffer(self, stream_buffer: StreamBuffer) -> None:
        """
        Coroutine used as callback for the audio bus listener to receive audio data chunks
        :param stream_buffer: the `StreamBuffer`s audio data chunk passed by audio service
        """
        if not self._recording:
            self.__log.debug('Cannot record audio buffer chunk: recording is not active!')
            return
        if not self._time_frames:
            self.__log.warning('Received a buffer for recording but no time frames exist')
            return
        for time_frame in reversed(self._time_frames):
            if time_frame.start_time <= stream_buffer.start_time:
                break
        else:
            self.__log.warning('Couldn\'t find appropriate time frame for stream buffer.'
                               'Buffer too old or discarded time frame')
            return
        time_frame.buffers.append(stream_buffer)  # pylint: disable=undefined-loop-variable
