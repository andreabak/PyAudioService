"""Module with common functions and classes"""

from abc import ABC, abstractmethod
from threading import Event, Thread
from typing import TypeVar, Sequence, Iterator


ST = TypeVar('ST')


def chunked(seq: Sequence[ST], n: int) -> Iterator[ST]:
    """Yield successive n-sized chunks from seq."""
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


class BackgroundService(ABC):
    """
    An abstract base class that implements basic interface and functionality for a background threaded service
    """
    def __init__(self):
        self._stop_event: Event = Event()
        self._thread: Thread = self._create_thread()

    @abstractmethod
    def _create_thread(self) -> Thread:
        """
        Abstract method to be overridden with a function that builds the thread
        :return: An initialized `Thread` instance
        """
        ...

    def start(self) -> None:
        """
        Starts the background thread
        """
        self._thread.start()

    def stop(self, wait: bool = True) -> None:
        """
        Stops the background thread and waits for it to finish.
        :param wait: If True, waits until the thread is stopped
        """
        self._stop_event.set()
        if wait:
            self._thread.join()

    def __enter__(self) -> None:
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
