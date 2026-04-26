"""Tiny listener-list helper.

A handful of services in Notekeeper expose multiple "subscription
channels" — segments, level, warnings, status, chunk events, note
saved, health snapshots, and so on — and every channel needs the same
three methods (``add``, ``remove``, ``notify``) plus identical
exception swallowing inside the loop.

This generic class collapses that boilerplate so a service just owns
one ``ListenerList`` per channel and writes ``self._foo.notify(value)``
when the event happens. The listener lock is internal so callers don't
have to think about whether they're emitting from the audio callback,
the asyncio worker, or the GUI thread.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Generic, Iterable, TypeVar

T = TypeVar("T")

log = logging.getLogger(__name__)


class ListenerList(Generic[T]):
    """A thread-safe list of single-argument callbacks.

    ``notify(value)`` invokes every registered callback in registration
    order. Listeners are called outside the lock so a slow listener
    can't block ``add`` / ``remove`` calls from another thread, and an
    exception in one listener is logged but doesn't suppress the
    others.
    """

    __slots__ = ("_listeners", "_lock", "_label")

    def __init__(self, *, label: str = "listener") -> None:
        self._listeners: list[Callable[[T], None]] = []
        self._lock = threading.Lock()
        # Only used in the log line on listener exceptions — keeps the
        # traceback grepable without forcing every channel through a
        # logger of its own.
        self._label = label

    def add(self, listener: Callable[[T], None]) -> None:
        with self._lock:
            self._listeners.append(listener)

    def remove(self, listener: Callable[[T], None]) -> None:
        with self._lock:
            try:
                self._listeners.remove(listener)
            except ValueError:
                pass  # idempotent — caller doesn't need to track membership

    def notify(self, value: T) -> None:
        with self._lock:
            snapshot: list[Callable[[T], None]] = list(self._listeners)
        for cb in snapshot:
            try:
                cb(value)
            except Exception:
                log.exception("%s listener raised", self._label)

    def __len__(self) -> int:
        with self._lock:
            return len(self._listeners)

    def __iter__(self) -> Iterable[Callable[[T], None]]:
        with self._lock:
            return iter(list(self._listeners))

    def clear(self) -> None:
        with self._lock:
            self._listeners.clear()
