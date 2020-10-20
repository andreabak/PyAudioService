import asyncio
import audioop
import subprocess
import time
from collections import deque
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field as dataclass_field
from threading import Event
from typing import List, MutableMapping, Any, Optional, Union

import ffmpeg
import numpy as np

from ..logger import custom_log
from .audio import StreamBuffer, PCMFormat, DEFAULT_FORMAT, PCMSampleFormat, AudioService, CHUNK_FRAMES


__all__ = [
    'AudioRecorder'
]


@dataclass
class StreamBuffersTimeFrame:
    start_time: float = time.monotonic()
    buffers: List[StreamBuffer] = dataclass_field(default_factory=list)


@custom_log(component='RECORDING')
class AudioRecorder:
    FRAMES_DELAY: int = 10
    INTERNAL_FORMAT: PCMFormat = PCMFormat(rate=DEFAULT_FORMAT.rate,
                                           sample_fmt=PCMSampleFormat.float32,
                                           channels=DEFAULT_FORMAT.channels)
    # DEFAULT_ENCODER_OPTIONS: MutableMapping[str, Any] = {}
    DEFAULT_ENCODER_OPTIONS: MutableMapping[str, Any] = {'c:a': 'aac', 'q:a': '192k'}

    def __init__(self, audio_service: AudioService, out_file_path: str, source_buses: Union[List[str], str],
                 pcm_format: Optional[PCMFormat] = None,
                 encoder_options: Optional[MutableMapping[str, Any]] = None):
        self._audio_service: AudioService = audio_service
        self._out_file_path: str = out_file_path
        if not source_buses:
            raise ValueError('Must specify at least one source bus for recording')
        if isinstance(source_buses, str):
            source_buses = [source_buses]
        self._source_buses: List[str] = source_buses
        if pcm_format is None:
            pcm_format = DEFAULT_FORMAT
        self._pcm_format: PCMFormat = pcm_format
        if encoder_options is None:
            encoder_options = self.DEFAULT_ENCODER_OPTIONS
        self._encoder_options: MutableMapping[str, Any] = encoder_options

        self._time_frames: deque[StreamBuffersTimeFrame] = deque()
        self._frame_size: int = CHUNK_FRAMES
        self._ffmpeg_process: Optional[subprocess.Popen] = None
        self._recording: bool = False
        self._stop_event: Event = Event()

    @property
    def stop_event(self) -> Event:
        return self._stop_event

    def start(self) -> None:
        self._audio_service.ensure_running()
        self._audio_service.loop.create_task(self.start_recording())

    def stop(self) -> None:
        self._stop_event.set()

    @contextmanager
    def record(self) -> None:
        self.start()
        try:
            yield
        finally:
            self.stop()

    async def start_recording(self):
        ffmpeg_spec: ffmpeg.Stream = ffmpeg.input('pipe:', **self.INTERNAL_FORMAT.ffmpeg_args) \
                                           .filter('alimiter', attack=10, release=20) \
                                           .output(self._out_file_path,
                                                   **self._pcm_format.ffmpeg_args_nofmt, **self._encoder_options)
        ffmpeg_args: List[str] = ffmpeg.compile(ffmpeg_spec, 'ffmpeg', overwrite_output=True)
        self._ffmpeg_process: subprocess.Popen = subprocess.Popen(args=ffmpeg_args, bufsize=0, text=False,
                                                                  stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            await self._recording_loop()
        except SystemExit:
            pass
        await self._time_frames_cleanup()
        while True:
            self.__log.debug('Waiting for ffmpeg to stop')
            ffmpeg_retcode: Optional[int] = self._ffmpeg_process.poll()
            if ffmpeg_retcode is None:
                await asyncio.sleep(0.1)
            # TODO: Failsafe to just end?

    async def _recording_loop(self):
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

                    if self._stop_event.is_set():
                        raise SystemExit

                    if len(self._time_frames) > self.FRAMES_DELAY:
                        await self._save_time_frame(self._time_frames.popleft())

                    sleep_delay: float = current_tf_time - time.monotonic() + time_step
                    if (time_shift := time_step - sleep_delay) > time_step:
                        self.__log.debug(f'Got some {time_shift*1000:.2f}ms of time shift on recording loop')
                    await asyncio.sleep(sleep_delay)
                    last_tf_time = current_tf_time
        finally:
            self._recording = False

    async def _time_frames_cleanup(self):
        while self._time_frames:
            await self._save_time_frame(self._time_frames.popleft())
        self._ffmpeg_process.stdin.close()

    async def _save_time_frame(self, time_frame: StreamBuffersTimeFrame) -> None:
        # TODO: Try bake calculations beforehand somewhere, without making code unreadable
        internal_fmt_channels = self.INTERNAL_FORMAT.channels
        internal_fmt_numpy = self.INTERNAL_FORMAT.sample_fmt.numpy
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
        # TODO: Detect if stdin.write fails, in case try to retrieve stderr and print it. Do we need async subprocess?
        self._ffmpeg_process.stdin.write(out_buffer)
        # while self._ffmpeg_process.stderr.readable()
        # while stderr_buf := self._ffmpeg_process.stderr.read(1024):
        #     print(stderr_buf, end='')
        # print()

    async def record_buffer(self, stream_buffer: StreamBuffer) -> None:
        # print(f'Recorder got {stream_buffer.size}bytes, {stream_buffer.frames} frames, '
        #       f'{stream_buffer.pcm_format}, '
        #       f'starts at {stream_buffer.start_time:.6f}, ends at {stream_buffer.end_time:.6f}')
        if not self._recording:
            self.__log.warning('Cannot record audio buffer chunk: recording is not active!')
        for time_frame in reversed(self._time_frames):
            if time_frame.start_time <= stream_buffer.start_time:
                break
        else:
            self.__log.warning('Couldn\'t find appropriate time frame for stream buffer.'
                               'Buffer too old or discarded time frame')
            return
        time_frame.buffers.append(stream_buffer)
