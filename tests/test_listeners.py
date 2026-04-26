"""``ListenerList`` — the small helper SessionManager uses for every channel."""

from __future__ import annotations

import threading

import pytest

from app.utils.listeners import ListenerList


def test_add_and_notify_invokes_each_listener_in_order():
    received: list[tuple[str, int]] = []
    channel: ListenerList[int] = ListenerList()
    channel.add(lambda v: received.append(("a", v)))
    channel.add(lambda v: received.append(("b", v)))

    channel.notify(7)
    channel.notify(8)

    assert received == [("a", 7), ("b", 7), ("a", 8), ("b", 8)]


def test_remove_listener_stops_emissions():
    received: list[int] = []

    def cb(v: int) -> None:
        received.append(v)

    channel: ListenerList[int] = ListenerList()
    channel.add(cb)
    channel.notify(1)
    channel.remove(cb)
    channel.notify(2)

    assert received == [1]


def test_remove_unknown_listener_is_a_noop():
    channel: ListenerList[int] = ListenerList()
    channel.remove(lambda v: None)  # never added — should not raise


def test_failing_listener_does_not_block_others(caplog):
    seen: list[int] = []

    def boom(_v: int) -> None:
        raise RuntimeError("kaboom")

    channel: ListenerList[int] = ListenerList(label="sample")
    channel.add(boom)
    channel.add(seen.append)

    with caplog.at_level("ERROR"):
        channel.notify(42)

    # The exception was swallowed and the second listener still ran.
    assert seen == [42]
    assert any("sample listener raised" in rec.message for rec in caplog.records)


def test_label_appears_in_log_message(caplog):
    channel: ListenerList[int] = ListenerList(label="my-channel")
    channel.add(lambda v: 1 / 0)
    with caplog.at_level("ERROR"):
        channel.notify(1)
    assert any("my-channel listener raised" in rec.message for rec in caplog.records)


def test_clear_removes_all_listeners():
    received: list[int] = []
    channel: ListenerList[int] = ListenerList()
    channel.add(received.append)
    channel.add(received.append)

    channel.clear()
    channel.notify(99)

    assert received == []
    assert len(channel) == 0


def test_len_matches_number_of_listeners():
    channel: ListenerList[int] = ListenerList()
    assert len(channel) == 0
    cb1, cb2 = (lambda v: None), (lambda v: None)
    channel.add(cb1)
    channel.add(cb2)
    assert len(channel) == 2
    channel.remove(cb1)
    assert len(channel) == 1


def test_notify_is_thread_safe_under_concurrent_add():
    """A listener registered mid-notify must not corrupt the iteration."""
    channel: ListenerList[int] = ListenerList()
    received: list[int] = []
    channel.add(received.append)

    def add_more(_v: int) -> None:
        # Add another listener from inside notify — should not deadlock.
        channel.add(lambda x: received.append(-x))

    channel.add(add_more)
    channel.notify(10)
    # First notify: ``received.append`` ran; ``add_more`` ran (added a third
    # listener but the snapshot was already taken).
    assert received == [10]

    channel.notify(20)
    assert received == [10, 20, -20]


def test_notify_does_not_hold_the_lock_during_callback():
    """Slow listeners must not block ``add`` from another thread."""
    channel: ListenerList[int] = ListenerList()
    started = threading.Event()
    blocker_release = threading.Event()
    add_succeeded = threading.Event()

    def slow_listener(_v: int) -> None:
        started.set()
        blocker_release.wait(timeout=2.0)

    channel.add(slow_listener)

    def fire_notify() -> None:
        channel.notify(1)

    notify_thread = threading.Thread(target=fire_notify)
    notify_thread.start()
    assert started.wait(timeout=1.0)

    # While the slow listener is parked, ``add`` from this thread must
    # complete promptly — this is the contract the SessionManager relies on.
    def add_from_main() -> None:
        channel.add(lambda v: None)
        add_succeeded.set()

    add_thread = threading.Thread(target=add_from_main)
    add_thread.start()
    assert add_succeeded.wait(timeout=1.0), (
        "add() blocked while a listener was running — notify is holding the lock"
    )

    blocker_release.set()
    notify_thread.join(timeout=2.0)
    add_thread.join(timeout=2.0)
