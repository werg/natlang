import fcntl
import hashlib
import multiprocessing
import time

from scripts.coreweave.private_export_guard import sandbox_upload_lease


def _old_reader(path, ready, release):
    with open(path, 'a+b') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        ready.set()
        release.wait(15)


def _put(directory, entered, begin):
    begin.wait(10)
    with sandbox_upload_lease(directory, 'sandbox', verified_legacy_reader=lambda stream: True):
        entered.set()


def test_completed_legacy_get_does_not_hold_new_puts_and_puts_remain_serial(tmp_path):
    ctx = multiprocessing.get_context('fork')
    ready, release, entered = ctx.Event(), ctx.Event(), ctx.Event()
    legacy = tmp_path / ('sandbox-upload-' + hashlib.sha256(b'sandbox').hexdigest() + '.lock')
    reader = ctx.Process(target=_old_reader, args=(str(legacy), ready, release))
    reader.start()
    try:
        assert ready.wait(5)
        begin = ctx.Event()
        put = ctx.Process(target=_put, args=(tmp_path, entered, begin))
        put.start()
        with sandbox_upload_lease(tmp_path, 'sandbox', verified_legacy_reader=lambda stream: True):
            begin.set()
            assert not entered.wait(.2), 'a second PUT must wait for the active PUT'
        assert entered.wait(5), 'a completed legacy GET must not block the next PUT'
        put.join(5)
        assert put.exitcode == 0 and reader.is_alive()
    finally:
        release.set()
        reader.join(5)
        if reader.is_alive():
            reader.kill()
            reader.join()


def test_unproven_legacy_holder_still_blocks_put(tmp_path):
    ctx = multiprocessing.get_context('fork')
    ready, release, entered = ctx.Event(), ctx.Event(), ctx.Event()
    legacy = tmp_path / ('sandbox-upload-' + hashlib.sha256(b'sandbox').hexdigest() + '.lock')
    reader = ctx.Process(target=_old_reader, args=(str(legacy), ready, release))
    reader.start()
    def unproven():
        with sandbox_upload_lease(tmp_path, 'sandbox', verified_legacy_reader=lambda stream: False):
            entered.set()
    put = ctx.Process(target=unproven)
    try:
        assert ready.wait(5)
        put.start()
        assert not entered.wait(.2)
        release.set()
        assert entered.wait(5)
        put.join(5)
        assert put.exitcode == 0
    finally:
        release.set()
        reader.join(5)
        if put.is_alive():
            put.kill()
            put.join()
