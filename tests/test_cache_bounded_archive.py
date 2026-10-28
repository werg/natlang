import hashlib
import io
import os
import tarfile

from scripts.coreweave.cache_bounded_archive import CacheBoundedWriter, open_recovery_tar


def test_archive_write_syncs_bounded_windows_and_final_tail(tmp_path, monkeypatch):
    sync = os.fdatasync
    calls = []
    def record(fd):
        calls.append(os.lseek(fd, 0, os.SEEK_CUR))
        sync(fd)
    monkeypatch.setattr(os, 'fdatasync', record)
    content = bytes(range(251)) * 100
    path = tmp_path / 'data'
    with CacheBoundedWriter(path, window_bytes=4096) as writer:
        assert writer.write(content) == len(content)
    assert path.read_bytes() == content
    assert calls[-1] == len(content)
    assert all(0 < b - a <= 4096 for a, b in zip([0] + calls, calls))


def test_tar_preserves_file_bytes_and_releases_source_cache(tmp_path, monkeypatch):
    content = bytes(range(253)) * 1000
    source = tmp_path / 'source'
    source.write_bytes(content)
    advise = os.posix_fadvise
    reads = []
    def record(fd, start, size, mode):
        if os.fstat(fd).st_ino == source.stat().st_ino:
            reads.append((start, size))
        advise(fd, start, size, mode)
    monkeypatch.setattr(os, 'posix_fadvise', record)
    archive = tmp_path / 'recovery.tar'
    with open_recovery_tar(archive) as tar:
        tar.add(source, arcname='source')
        info = tarfile.TarInfo('metadata'); info.size = 2
        tar.addfile(info, io.BytesIO(b'{}'))
    with tarfile.open(archive) as tar:
        assert hashlib.sha256(tar.extractfile('source').read()).digest() == hashlib.sha256(content).digest()
        assert tar.extractfile('metadata').read() == b'{}'
    assert sum(size for _, size in reads) == len(content)
