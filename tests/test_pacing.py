import random
import threading
import time

import pytest

from arxiv_agent.clients.pacing import LocalPacer, SharedPacer


def overlaps(pacers, rounds: int = 5) -> int:
    # Runs turns from several threads; counts moments with two turns at once.
    inside, worst = 0, 0
    lock = threading.Lock()

    def take_turns(pacer):
        nonlocal inside, worst
        for _ in range(rounds):
            with pacer.turn():
                with lock:
                    inside += 1
                    worst = max(worst, inside)
                time.sleep(0.01)
                with lock:
                    inside -= 1

    threads = [threading.Thread(target=take_turns, args=(p,)) for p in pacers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return worst


def test_threads_sharing_a_local_pacer_take_turns():
    pacer = LocalPacer(interval=0)

    assert overlaps([pacer, pacer, pacer]) == 1


@pytest.fixture
def key():
    # A lock key of its own, so a real worker can't hold up the test.
    return random.randint(1, 2**31)


def test_processes_on_one_database_take_turns(connect, key):
    # Two connections stand for two processes; threads share each of them.
    first = SharedPacer(connect(), interval=0, key=key)
    second = SharedPacer(connect(), interval=0, key=key)

    assert overlaps([first, first, second, second]) == 1


def test_the_next_request_waits_out_the_interval_from_the_last_one(connect, key):
    waits: list[float] = []
    first = SharedPacer(connect(), interval=3.0, sleep=waits.append, key=key)
    second = SharedPacer(connect(), interval=3.0, sleep=waits.append, key=key)

    with first.turn():
        pass
    with second.turn():  # another process, right after
        pass

    [wait] = waits
    assert 2.5 < wait <= 3.0
