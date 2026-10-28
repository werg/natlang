"""Tar recovery files without accumulating their entire contents in page cache."""
from contextlib import contextmanager
import io
import os
import tarfile

from scripts.coreweave.private_export_guard import CacheBoundedReader


class CacheBoundedWriter(io.FileIO):
    def __init__(self, path, *, window_bytes=16 << 20):
        if type(window_bytes) is not int or window_bytes <= 0:
            raise ValueError('write window must be a positive byte count')
        super().__init__(path, 'x')
        self.window_bytes = window_bytes
        self._released = 0

    def _release(self):
        end = self.tell()
        if end > self._released:
            os.fdatasync(self.fileno())
            os.posix_fadvise(self.fileno(), self._released, end - self._released,
                            os.POSIX_FADV_DONTNEED)
            self._released = end

    def write(self, data):
        view = memoryview(data).cast('B')
        written = 0
        while written < len(view):
            remaining = self.window_bytes - (self.tell() - self._released)
            count = super().write(view[written:written + remaining])
            if not count:
                raise OSError('archive write made no progress')
            written += count
            if self.tell() - self._released >= self.window_bytes:
                self._release()
        return written

    def close(self):
        if not self.closed:
            try:
                self._release()
            finally:
                super().close()


class _CacheBoundedTar(tarfile.TarFile):
    def addfile(self, tarinfo, fileobj=None):
        try:
            fd = fileobj.fileno() if fileobj is not None else None
        except (AttributeError, io.UnsupportedOperation):
            fd = None
        if fd is None:
            return super().addfile(tarinfo, fileobj)
        # The duplicate shares the source offset but owns only its own descriptor.
        with CacheBoundedReader(os.dup(fd)) as reader:
            try:
                return super().addfile(tarinfo, reader)
            finally:
                reader._after_read(True)


@contextmanager
def open_recovery_tar(path):
    with CacheBoundedWriter(path) as output:
        with _CacheBoundedTar.open(fileobj=output, mode='w', dereference=True) as archive:
            yield archive
