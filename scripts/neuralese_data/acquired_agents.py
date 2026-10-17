"""Pinned new agent snapshots through the existing trajectory representation."""
from __future__ import annotations

import json
import gzip
import mmap
import re
from pathlib import Path

from .common import Reject
from .records import group_key

_D14_JSON_STRUCTURE = re.compile(rb'["{}\[\]]')
_D14_JSON_STRING_MARKER = re.compile(rb'["\\]')
_D14_JSON_SCALAR_END = re.compile(rb'[,}\]]')

ACQUIRED_FILES = None  # Optional exact receipt-file shard, never an implicit row limit.
ACQUIRED_RECEIPTS = {}  # Explicit immutable selected-file snapshots; never the mutable download journal.
_ACQUISITION_CACHE = {}
_ACQUISITION_FILE_INDEX = {}

# A source ref is normalized through the existing shared Trajectory adapter.
# Keep this lookup beside those adapters so export workers do not maintain a
# second, incomplete copy of the ref-to-corpus contract.
FULL_TRAJECTORY_CORPORA = {
    'D13': 'team-cooperation-acquired',
    'D14': 'qwen-cooperation-acquired',
    'D17': 'cooperdata-acquired',
    'D19': 'agent-collusion-raw-acquired',
    'D20': 'swe-chat-acquired',
    'D22': 'open-swe-acquired',
    'D23': 'swe-zero-acquired',
    'D24': 'miroverse-acquired',
    'D25': 'swe-rebench-acquired',
    'D26': 'toucan-acquired',
    'D27': 'toolmind-acquired',
    'D28': 'nemotron-agentic-acquired',
    'D32': 'openresearcher-structured-acquired',
    'D35': 'nemotron-terminal-acquired',
    'D36': 'coderforge-acquired',
    'D37': 'orchard-acquired',
    'D38': 'ultradata-acquired',
}


def full_trajectory_corpus(ref):
    """Resolve a pinned source ref through the canonical complete-session adapter."""
    from . import agents
    try:
        name = FULL_TRAJECTORY_CORPORA[ref]
    except KeyError as error:
        raise Reject('no complete-session adapter is registered for ' + str(ref)) from error
    if name not in agents.CORPORA:
        raise Reject('complete-session adapter is unavailable for ' + str(ref))
    return name, agents.CORPORA[name]


def supported_acquired_file(name):
    """One shared pinned-file predicate for conversion and export inventories."""
    return str(name).endswith(('.parquet','.jsonl','.jsonl.gz','.jsonl.zst','.json','.json.gz','.tar.gz'))


def _d14_json_projection_value_end(buffer, start, size):
    """Find a JSON value boundary without decoding strings that may be discarded."""
    while start < size and buffer[start] in (9, 10, 13, 32):
        start += 1
    if start >= size:
        raise Reject('truncated D14 JSON value')
    token = buffer[start]
    if token == 34:
        return _d14_json_projection_string_end(buffer, start, size)
    if token not in (123, 91):
        match = _D14_JSON_SCALAR_END.search(buffer, start, size)
        end = size if match is None else match.start()
        while end > start and buffer[end - 1] in (9, 10, 13, 32):
            end -= 1
        if end == start:
            raise Reject('empty D14 JSON scalar')
        return end
    stack = [125 if token == 123 else 93]
    cursor = start + 1
    while stack:
        match = _D14_JSON_STRUCTURE.search(buffer, cursor, size)
        if match is None:
            raise Reject('unterminated D14 JSON container')
        position = match.start()
        byte = buffer[position]
        if byte == 34:
            cursor = _d14_json_projection_string_end(buffer, position, size)
            continue
        if byte in (123, 91):
            stack.append(125 if byte == 123 else 93)
        else:
            if byte != stack[-1]:
                raise Reject('mismatched D14 JSON container')
            stack.pop()
        cursor = position + 1
    return cursor


def _d14_json_projection_string_end(buffer, start, size):
    if start >= size or buffer[start] != 34:
        raise Reject('expected D14 JSON string')
    cursor = start + 1
    while cursor < size:
        match = _D14_JSON_STRING_MARKER.search(buffer, cursor, size)
        if match is None:
            break
        byte = buffer[match.start()]
        if byte == 34:
            return match.start() + 1
        # In JSON every escape consumes exactly one following byte (including
        # the first byte of a unicode escape); that byte cannot terminate the
        # string. Searching again after it preserves linear progress without
        # carrying escape state across ordinary bytes.
        cursor = match.start() + 2
    raise Reject('unterminated D14 JSON string')


class _D14IgnoredToolExtraProjection:
    """Materialize the exact D14 document while skipping unused tool `extra` values."""

    def __init__(self, buffer):
        self.buffer = buffer
        self.size = len(buffer)
        self.omitted_fields = 0
        self.omitted_value_bytes = 0

    def _space(self, position):
        while position < self.size and self.buffer[position] in (9, 10, 13, 32):
            position += 1
        return position

    def _string(self, position):
        end = _d14_json_projection_string_end(self.buffer, position, self.size)
        try:
            return json.loads(self.buffer[position:end]), end
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise Reject('malformed D14 JSON string') from error

    def _object_fields(self, position):
        position = self._space(position + 1)
        fields = []
        if position < self.size and self.buffer[position] == 125:
            return fields, position + 1
        while position < self.size:
            key, key_end = self._string(position)
            position = self._space(key_end)
            if position >= self.size or self.buffer[position] != 58:
                raise Reject('malformed D14 JSON object field')
            value_start = self._space(position + 1)
            value_end = _d14_json_projection_value_end(self.buffer, value_start, self.size)
            fields.append((key, value_start, value_end))
            position = self._space(value_end)
            if position < self.size and self.buffer[position] == 44:
                position = self._space(position + 1)
                continue
            if position < self.size and self.buffer[position] == 125:
                return fields, position + 1
            raise Reject('malformed D14 JSON object delimiter')
        raise Reject('unterminated D14 JSON object')

    def value(self, position, context='ordinary'):
        position = self._space(position)
        if position >= self.size:
            raise Reject('truncated D14 JSON document')
        token = self.buffer[position]
        if token == 34:
            value, end = self._string(position)
            return value, end
        if token == 123:
            fields, end = self._object_fields(position)
            role = None
            for key, start, stop in fields:
                if key == 'role':
                    try:
                        role = json.loads(self.buffer[start:stop])
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        role = None
            result = {}
            for key, start, stop in fields:
                if key == 'extra' and context == 'message' and role == 'tool':
                    self.omitted_fields += 1
                    self.omitted_value_bytes += stop - start
                    continue
                child_context = 'ordinary'
                if context in ('root', 'row') and key == 'messages':
                    child_context = 'messages'
                elif context in ('root', 'row') and key == 'segments':
                    child_context = 'segments'
                elif context == 'segment' and key == 'messages':
                    child_context = 'messages'
                result[key], _ = self.value(start, child_context)
            return result, end
        if token == 91:
            result = []
            cursor = self._space(position + 1)
            if cursor < self.size and self.buffer[cursor] == 93:
                return result, cursor + 1
            while cursor < self.size:
                child_context = 'ordinary'
                if context == 'messages':
                    child_context = 'message'
                elif context == 'segments':
                    child_context = 'segment'
                elif context == 'root':
                    child_context = 'row'
                item, end = self.value(cursor, child_context)
                result.append(item)
                cursor = self._space(end)
                if cursor < self.size and self.buffer[cursor] == 44:
                    cursor = self._space(cursor + 1)
                    continue
                if cursor < self.size and self.buffer[cursor] == 93:
                    return result, cursor + 1
                raise Reject('malformed D14 JSON array delimiter')
            raise Reject('unterminated D14 JSON array')
        end = _d14_json_projection_value_end(self.buffer, position, self.size)
        try:
            return json.loads(self.buffer[position:end]), end
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise Reject('malformed D14 JSON scalar') from error


def _iter_d14_full_trajectory_json(path, source_entry):
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    path = Path(path)
    if (path.stat().st_size != source_entry.get('bytes')
            or sha256_file_hex_cached(path) != source_entry.get('sha256')):
        raise Reject('D14 projected source differs from its acquisition binding')
    with path.open('rb') as stream:
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as buffer:
            parser = _D14IgnoredToolExtraProjection(buffer)
            value, end = parser.value(0, 'root')
            if parser._space(end) != len(buffer):
                raise Reject('trailing bytes in D14 full-trajectory JSON')
            projection = {
                'schema': 'natlang.d14-ignored-tool-extra-projection/1',
                'parser': 'receipt-bound-mmap-json-projector-v1',
                'source_file_sha256': source_entry['sha256'],
                'source_file_bytes': source_entry['bytes'],
                'omitted_message_extra_fields': parser.omitted_fields,
                'omitted_raw_value_bytes': parser.omitted_value_bytes,
                'discarded_value_validation': 'balanced JSON string/container boundaries; skipped contents are not independently decoded',
                'training_admission': False,
            }
            if isinstance(value, list):
                for index, row in enumerate(value):
                    if isinstance(row, dict):
                        row['_natlang_source_projection'] = projection
                    yield index, row
            else:
                if isinstance(value, dict):
                    value['_natlang_source_projection'] = projection
                yield 0, value


def iter_acquired_file_rows(path, *, d14_projection_entry=None):
    """Yield original integer rows; blank JSONL lines keep their positions."""
    path=Path(path)
    if str(path).endswith('.tar.gz'):
        from .acquired_archives import iter_archive_rows
        yield from iter_archive_rows(path)
    elif path.suffix=='.parquet':
        from . import agents
        yield from agents._parquet_groups(path)
    elif d14_projection_entry is not None:
        if not path.name.endswith('_full_traj.json'):
            raise Reject('D14 ignored-extra projection only supports full-trajectory JSON')
        yield from _iter_d14_full_trajectory_json(path, d14_projection_entry)
    elif str(path).endswith(('.json','.json.gz')):
        opener=gzip.open if str(path).endswith('.gz') else open
        with opener(path,'rt',encoding='utf-8') as stream:
            value=json.load(stream)
        if isinstance(value,list):yield from enumerate(value)
        else:yield 0,value
    elif supported_acquired_file(path):
        from natlang_neuralese.common.jsonio import open_jsonl_text
        opener=gzip.open if str(path).endswith('.gz') else open_jsonl_text
        options={'mode':'rt','encoding':'utf-8'} if str(path).endswith('.gz') else {}
        with opener(path,**options) as stream:
            first=stream.readline()
            # Some released SWE transcript files use a .jsonl suffix while
            # containing one pretty-printed session document. Recognize only
            # an isolated top-level opener; ordinary one-object-per-line JSONL
            # remains on the streaming path below.
            if first.strip() in ('{','['):
                document=json.loads(first+stream.read())
                if isinstance(document,list):
                    yield from enumerate(document)
                else:
                    yield 0,document
                return
            if first.strip():
                yield 0,json.loads(first)
            for number,line in enumerate(stream):
                if line.strip():yield number+1,json.loads(line)
    else:
        raise Reject('unsupported acquired source file: '+path.name)


def iter_acquired_file_row_ids(path):
    """Enumerate source row positions without materializing unused Parquet columns."""
    path=Path(path)
    if path.suffix=='.parquet':
        import pyarrow.parquet as pq
        # The file's immutable byte digest is checked by the caller before this
        # metadata is trusted. Parquet row groups provide exact positional
        # accounting without decoding any column payload.
        yield from range(pq.ParquetFile(str(path)).metadata.num_rows)
        return
    if str(path).endswith(('.jsonl','.jsonl.gz','.jsonl.zst')):
        import gzip
        from natlang_neuralese.common.jsonio import open_jsonl_text
        opener=gzip.open if str(path).endswith('.gz') else open_jsonl_text
        options={'mode':'rt','encoding':'utf-8'} if str(path).endswith('.gz') else {}
        with opener(path,**options) as stream:
            first=stream.readline()
            if first.strip() in ('{','['):
                document=json.loads(first+stream.read())
                yield from range(len(document)) if isinstance(document,list) else (0,)
                return
            if first.strip():yield 0
            for number,line in enumerate(stream):
                if line.strip():yield number+1
        return
    raise Reject('metadata-only row accounting supports Parquet and JSONL files only')


def _bound_acquired_path(root,entry,receipt=None):
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    root=Path(root).resolve();relative=Path(entry['path'])
    materialized_root=root
    if receipt is not None:
        labels=receipt.get('materialized_roots') or {}
        locations=receipt.get('materialized_files') or {}
        label=locations.get(entry['path'])
        if label is not None:
            candidate=labels.get(label)
            if not isinstance(candidate,str) or not Path(candidate).is_absolute():
                raise Reject('acquired source location lacks an exact absolute root: '+str(relative))
            try:
                materialized_root=Path(candidate).resolve(strict=True)
            except OSError as error:
                raise Reject('acquired source location root is unavailable: '+str(relative)) from error
            if not materialized_root.is_dir():
                raise Reject('acquired source location root is not a directory: '+str(relative))
    try:
        path=(materialized_root/relative).resolve(strict=True)
    except OSError as error:
        raise Reject('acquired source file differs: '+str(relative)) from error
    if not path.is_relative_to(materialized_root) or path.stat().st_size!=entry['bytes']:
        raise Reject('acquired source file differs: '+str(relative))
    if sha256_file_hex_cached(path)!=entry['sha256']:
        raise Reject('acquired source digest differs: '+str(relative))
    return path


def cooperation_trajectory_file(ref,name,known_files):
    """Prefer the complete representation, retaining agents without one."""
    name=str(name)
    if not Path(name).name.startswith('agent') or not name.endswith('_traj.json'):return False
    if ref=='D14' and not name.endswith('_full_traj.json'):
        return name.removesuffix('_traj.json')+'_full_traj.json' not in known_files
    return ref in ('D13','D14')


def source_json_value(value):
    """Addressable Arrow metadata without dropping dates, bytes or NaN fields."""
    from datetime import date,datetime
    import base64
    import math
    if isinstance(value,dict):return {key:source_json_value(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [source_json_value(item) for item in value]
    if isinstance(value,(date,datetime)):
        return {'$source_type':type(value).__name__,'iso8601':value.isoformat()}
    if isinstance(value,bytes):return {'$source_type':'bytes','base64':base64.b64encode(value).decode('ascii')}
    if isinstance(value,float) and not math.isfinite(value):
        return {'$source_type':'float','hex':value.hex()}
    if value is None or isinstance(value,(str,int,float,bool)):return value
    raise Reject('unsupported original Arrow metadata type: '+type(value).__name__)


def load_acquisition(ref):
    """Verify complete acquisitions or explicit byte-bound catalogue subsets."""
    from . import agents
    root = agents.EXPANSION_ROOT/ref
    receipt_path = Path(ACQUIRED_RECEIPTS.get(ref, root/'acquisition.json'))
    receipt_path = receipt_path.resolve(strict=True)
    root_key = str(root.resolve())
    stat = receipt_path.stat()
    cache_key = (ref, root_key, str(receipt_path), stat.st_dev, stat.st_ino,
                 stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    cached = _ACQUISITION_CACHE.get(cache_key)
    if cached is not None:
        archive_path, archive_stat = cached[1]
        try:
            current = archive_path.stat()
        except FileNotFoundError:
            current = None
        if current is not None and _stat_identity(current) == archive_stat:
            return cached[0]
        _ACQUISITION_CACHE.pop(cache_key, None)
    receipt = json.loads(receipt_path.read_text())
    receipt = validate_acquisition(ref, receipt, receipt_path, root=root)
    archive = receipt.get('source_archive') or {}
    archive_path = Path(archive['path']).resolve() if archive.get('path') else None
    if archive_path is not None and archive_path.is_file():
        archive_stat = archive_path.stat()
        archive_binding = (archive_path, _stat_identity(archive_stat))
    elif (archive_path is not None
            and receipt.get('schema') == 'natlang.external-source-recovery/1'
            and archive.get('verified_local_archive')):
        # This branch is available only after validate_acquisition verifies
        # both retained sidecars and every extracted source member.
        archive_binding = (receipt_path, _stat_identity(stat))
    elif archive_path is not None:
        # Keep the original fail-closed behavior for ordinary acquired sources
        # and for recovery receipts without the explicit pruned-archive proof.
        archive_stat = archive_path.stat()
        archive_binding = (archive_path, _stat_identity(archive_stat))
    else:
        archive_binding = (receipt_path, _stat_identity(stat))
    _ACQUISITION_CACHE[cache_key] = (receipt, archive_binding)
    return receipt


def _acquired_receipt_index(ref, receipt):
    """Index immutable receipt entries once for per-file conversion shards."""
    key = (ref, id(receipt))
    cached = _ACQUISITION_FILE_INDEX.get(key)
    if cached is not None and cached[0] is receipt:
        return cached[1]
    supported = []
    by_path = {}
    positions = {}
    for position, entry in enumerate(receipt['files']):
        name = entry['path']
        if supported_acquired_file(name):
            supported.append(entry)
            by_path[name] = entry
            positions[name] = position
    result = (frozenset(by_path), by_path, positions, supported)
    # Keep a strong reference: object ids may be reused after receipt eviction.
    _ACQUISITION_FILE_INDEX[key] = (receipt, result)
    return result


def _stat_identity(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def validate_acquisition(ref, receipt, receipt_path, *, root):
    """Shared prepublication and read-time acquisition verifier."""
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    root=Path(root);receipt_path=Path(receipt_path)
    if receipt.get('schema') == 'natlang.external-source-recovery/1':
        return validate_recovery_manifest(ref, receipt, receipt_path, root=root)
    if receipt.get('status') == 'acquired' and receipt.get('revision'):
        return receipt
    if (receipt.get('schema') != 'natlang.acquired-file-snapshot/1'
            or receipt.get('status') != 'acquired-file-snapshot'
            or receipt.get('source_complete') is not False or not receipt.get('revision')):
        raise Reject('source snapshot is not completely acquired and pinned: '+ref)
    binding = receipt.get('catalogue') or {}
    catalogue_path = receipt_path.parent / binding.get('path', '')
    if (not catalogue_path.is_file() or catalogue_path.stat().st_size != binding.get('bytes')
            or sha256_file_hex_cached(catalogue_path) != binding.get('sha256')):
        raise Reject('selected source snapshot catalogue binding differs')
    catalogue = json.loads(catalogue_path.read_text())
    sources = [source for source in catalogue.get('sources', []) if source.get('ref') == ref]
    if len(sources) != 1:
        raise Reject('selected source snapshot lacks unique original catalogue source')
    source = sources[0]
    if source.get('repo') != receipt.get('repo') or source.get('revision') != receipt['revision']:
        raise Reject('selected source snapshot differs from pinned catalogue identity')
    known = {entry['path']:entry for entry in source['files']}
    if len(known) != len(source['files']):
        raise Reject('duplicate original catalogue source path')
    selected = {entry['path']:entry for entry in receipt.get('files', [])}
    if not selected or len(selected) != len(receipt['files']) or not selected.keys() <= known.keys():
        raise Reject('invalid selected source file inventory')
    if sorted(receipt.get('pending_files', [])) != sorted(known.keys()-selected.keys()):
        raise Reject('selected source snapshot loses unacquired catalogue obligations')
    for name, entry in selected.items():
        expected = known[name]
        if entry.get('bytes') != expected.get('bytes') or not isinstance(entry.get('sha256'), str) or len(entry['sha256']) != 64:
            raise Reject('selected source file differs from catalogue size/digest contract')
        path = _bound_acquired_path(root, entry)
        if expected.get('lfs_sha256'):
            if entry['sha256'] != expected['lfs_sha256']:
                raise Reject('selected source file differs from pinned LFS digest')
        elif expected.get('blob_id'):
            from .acquired_snapshot import git_blob_hex
            if git_blob_hex(path) != expected['blob_id']:
                raise Reject('selected source file differs from pinned Git blob')
        else:
            raise Reject('selected source catalogue lacks immutable remote file identity')
    return receipt


def validate_recovery_manifest(ref, receipt, receipt_path, *, root):
    """Validate an explicitly recovered pinned source without impersonating its lost receipt.

    A recovery manifest binds the authenticated private archive, the pinned
    upstream identity, and a freshly hashed complete file inventory.  The
    unavailable historical acquisition receipt remains an explicit fact.
    """
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    import hashlib
    import tarfile
    root=Path(root).resolve();receipt_path=Path(receipt_path).resolve()
    if (receipt.get('ref') != ref or receipt.get('status') != 'recovered'
            or receipt.get('training_admission') is not False
            or not isinstance(receipt.get('repo'),str) or not receipt['repo']
            or not isinstance(receipt.get('revision'),str) or not receipt['revision']):
        raise Reject('recovered source identity or status is invalid: '+str(ref))
    history=receipt.get('historical_acquisition_receipt') or {}
    historical_sha=history.get('expected_sha256')
    stage_proof_sha=history.get('expected_source_stage_proof_sha256')
    archive=receipt.get('source_archive') or {}
    if (history.get('available') is not False
            or (historical_sha is not None and
                (not isinstance(historical_sha,str) or len(historical_sha) != 64
                 or any(char not in '0123456789abcdef' for char in historical_sha)))
            or (historical_sha is None and
                (not isinstance(stage_proof_sha,str) or len(stage_proof_sha) != 64
                 or any(char not in '0123456789abcdef' for char in stage_proof_sha)))
            or archive.get('verified') is not True
            or archive.get('private') is not True
            or type(archive.get('bytes')) is not int or archive['bytes'] <= 0
            or not isinstance(archive.get('sha256'),str)
            or len(archive['sha256']) != 64
            or any(char not in '0123456789abcdef' for char in archive['sha256'])
            or not isinstance(archive.get('run'),str)
            or not isinstance(archive.get('name'),str)
            or not isinstance(archive.get('path'),str)):
        raise Reject('recovered source lacks an exact archive and historical-receipt status')
    if receipt_path.parent.resolve() != root:
        raise Reject('recovery manifest is outside its bound source root')
    embedded = receipt.get('embedded_archive_manifest') or {}
    embedded_name = embedded.get('path')
    if (not isinstance(embedded_name, str) or not embedded_name
            or Path(embedded_name).is_absolute() or '..' in Path(embedded_name).parts
            or type(embedded.get('bytes')) is not int or embedded['bytes'] <= 0
            or not isinstance(embedded.get('sha256'), str)
            or len(embedded['sha256']) != 64
            or any(char not in '0123456789abcdef' for char in embedded['sha256'])):
        raise Reject('recovery manifest lacks an exact embedded source-stage proof')
    if historical_sha is None and stage_proof_sha != embedded['sha256']:
        raise Reject('unavailable historical receipt must bind the exact retained source-stage proof')
    archive_path=Path(archive.get('path','')).resolve()
    proof_bytes=None
    if archive_path.is_file():
        if archive_path.stat().st_size != archive['bytes'] or sha256_file_hex_cached(archive_path) != archive['sha256']:
            raise Reject('recovered source archive differs from its private-object binding')
        try:
            with tarfile.open(archive_path, 'r:*') as stream:
                member = stream.getmember(embedded_name)
                if not member.isfile() or member.size != embedded['bytes']:
                    raise Reject('embedded source-stage proof differs from its archive binding')
                proof_bytes = stream.extractfile(member).read()
        except (KeyError, OSError, tarfile.TarError) as error:
            raise Reject('recovered source archive lacks its bound source-stage proof') from error
    else:
        # A streamed private-object GET may be discarded after exact archive
        # hashing and member extraction. Retain the verified object receipt and
        # the archive's exact embedded inventory bytes beside the source; this
        # avoids keeping a second multi-gigabyte copy of the source data.
        local=archive.get('verified_local_archive') or {}
        verification_path=Path(local.get('verification_receipt_path',''))
        embedded_path=Path(embedded.get('local_path',''))
        try:
            verification_resolved=(root / verification_path).resolve(strict=True) if not verification_path.is_absolute() else verification_path.resolve(strict=True)
            embedded_resolved=(root / embedded_path).resolve(strict=True) if not embedded_path.is_absolute() else embedded_path.resolve(strict=True)
        except OSError as error:
            raise Reject('recovered source lacks retained private-archive verification sidecars') from error
        if (not verification_resolved.is_relative_to(root) or not embedded_resolved.is_relative_to(root)
                or not verification_resolved.is_file() or not embedded_resolved.is_file()):
            raise Reject('recovery sidecars escape the exact source root')
        verification_bytes=verification_resolved.read_bytes()
        if (hashlib.sha256(verification_bytes).hexdigest() != local.get('verification_receipt_sha256')
                or embedded_resolved.stat().st_size != embedded['bytes']
                or sha256_file_hex_cached(embedded_resolved) != embedded['sha256']):
            raise Reject('retained private-archive sidecar differs from its exact binding')
        try:
            verification=json.loads(verification_bytes)
            proof_bytes=embedded_resolved.read_bytes()
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise Reject('retained private-archive sidecar is malformed') from error
        if (verification.get('schema') != 'natlang.private-source-archive-local-verification/1'
                or verification.get('verified_private_object') is not True
                or verification.get('archive_retained') is not False
                or verification.get('run') != archive['run']
                or verification.get('name') != archive['name']
                or verification.get('bytes') != archive['bytes']
                or verification.get('sha256') != archive['sha256']
                or verification.get('embedded_manifest') != {
                    'path': embedded_name, 'bytes': embedded['bytes'],
                    'sha256': embedded['sha256'], 'schema': embedded.get('schema')}):
            raise Reject('retained receipt does not prove the exact verified private archive and embedded manifest')
    if hashlib.sha256(proof_bytes).hexdigest() != embedded['sha256']:
        raise Reject('embedded source-stage proof digest differs')
    try:
        proof = json.loads(proof_bytes)
    except json.JSONDecodeError as error:
        raise Reject('recovered source archive embedded proof is malformed') from error
    source_receipt = proof.get('acquisition') or {}
    if (proof.get('schema') != embedded.get('schema')
            or proof.get('schema') != 'natlang.pinned-public-source-archive/1'
            or source_receipt.get('schema') != 'natlang.external-source-acquisition/1'
            or source_receipt.get('ref') != ref
            or source_receipt.get('repo') != receipt.get('repo')
            or source_receipt.get('revision') != receipt.get('revision')
            or source_receipt.get('files') != receipt.get('files')):
        raise Reject('recovered source inventory is not bound to its retained archive proof')
    files=receipt.get('files')
    if not isinstance(files,list) or not files:
        raise Reject('recovered source has no complete file inventory')
    materialized_roots=receipt.get('materialized_roots') or {}
    materialized_files=receipt.get('materialized_files') or {}
    if not isinstance(materialized_roots,dict) or not isinstance(materialized_files,dict):
        raise Reject('recovered source locations must be explicit root and file mappings')
    resolved_roots={}
    for label,location in materialized_roots.items():
        if (not isinstance(label,str) or not label or not isinstance(location,str)
                or not Path(location).is_absolute()):
            raise Reject('recovered source location root binding is invalid')
        try:
            resolved=Path(location).resolve(strict=True)
        except OSError as error:
            raise Reject('recovered source location root is unavailable') from error
        if not resolved.is_dir():
            raise Reject('recovered source location root is not a directory')
        resolved_roots[label]=str(resolved)
    if (len(resolved_roots)!=len(materialized_roots)
            or len(set(resolved_roots.values()))!=len(resolved_roots)
            or any(not isinstance(name,str) or not isinstance(label,str)
                   or label not in resolved_roots for name,label in materialized_files.items())
            or set(resolved_roots)!=set(materialized_files.values())):
        raise Reject('recovered source file locations are invalid or ambiguous')
    seen=set()
    for entry in files:
        if not isinstance(entry,dict):
            raise Reject('recovered source inventory entry is invalid')
        name=entry.get('path');size=entry.get('bytes');digest=entry.get('sha256')
        path=Path(name) if isinstance(name,str) else Path('..')
        if (not isinstance(name,str) or path.is_absolute() or '..' in path.parts
                or name in seen or type(size) is not int or size < 0
                or not isinstance(digest,str) or len(digest) != 64
                or any(char not in '0123456789abcdef' for char in digest)):
            raise Reject('recovered source inventory contains an unsafe or duplicate entry')
        try:
            _bound_acquired_path(root,entry,receipt)
        except Reject as error:
            raise Reject('recovered source inventory file differs') from error
        seen.add(name)
    if not set(materialized_files)<=seen:
        raise Reject('recovered source locations name an unbound file')
    inventory_sha=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    expected=receipt.get('expected_source_inventory') or {}
    if (inventory_sha != receipt.get('file_inventory_sha256')
            or inventory_sha != expected.get('sha256')
            or type(expected.get('files')) is not int
            or type(expected.get('bytes')) is not int
            or len(files) != expected.get('files')
            or sum(entry['bytes'] for entry in files) != expected.get('bytes')):
        raise Reject('recovered source inventory differs from the retained source-stage binding')
    return receipt


def acquired_rows(ref, limit=None, *, metadata_only_files=frozenset()):
    from . import agents
    root = agents.EXPANSION_ROOT / ref
    receipt = load_acquisition(ref)
    known, by_path, positions, supported_entries = _acquired_receipt_index(ref, receipt)
    if ACQUIRED_FILES is not None and not ACQUIRED_FILES <= known:
        raise Reject('requested file is not a bound source in this receipt')
    if ACQUIRED_FILES is None:
        entries = supported_entries
    else:
        entries = sorted((by_path[name] for name in ACQUIRED_FILES), key=lambda entry: positions[entry['path']])
    index = 0
    for entry in entries:
        relative = Path(entry['path'])
        path=_bound_acquired_path(root,entry,receipt)
        metadata_only = str(relative) in metadata_only_files
        if metadata_only and path.suffix!='.parquet':
            raise Reject('metadata-only acquired row accounting requires a Parquet source')
        d14_projection_entry=(entry if ref=='D14' and entry['path'].endswith('_full_traj.json') else None)
        rows=((row_index,None) for row_index in iter_acquired_file_row_ids(path)) if metadata_only else iter_acquired_file_rows(path,d14_projection_entry=d14_projection_entry)
        for row_index, row in rows:
            if limit is not None and index >= limit:
                return
            index += 1
            yield receipt, f'{relative}:{row_index}', row


def _reported_session(ref, receipt, rid, row, messages, *, task_id, session_id, teacher, tools=None):
    from . import agents
    if not isinstance(messages, list) or not messages:
        raise Reject('missing original session messages')
    if not task_id or not session_id:
        raise Reject('missing original task/session identity')
    meta = row.get('metadata') or {}
    if isinstance(meta, str):
        meta = json.loads(meta)
    trajectory = agents.expansion_trajectory(ref, receipt, rid, {
        'messages': messages, 'tools': tools,
        'metadata': {**meta, 'trajectory_id': str(task_id), 'session_id': str(session_id),
                     'source_row_id': rid, 'original_model': teacher},
        '_expansion_split': 'test' if row.get('split') in ('test', 'validation', 'eval') else 'train',
    })
    trajectory.lineage['teacher'] = agents._teacher(teacher)
    instance = row.get('instance_id')
    if instance:
        trajectory.split_groups.append(group_key('swe-instance', instance))
        if instance in agents._bench_ids():
            trajectory.split = 'test'
    if row.get('repo'):
        trajectory.split_groups.append(group_key('repo', row['repo']))
    reported = row.get('resolved')
    trajectory.outcome = {'label': 'failed' if reported is False or reported == 0 else 'teacher',
        'checked': None, 'details': {'reported_resolved': reported,
                                   'origin': 'upstream_metadata', 'executed_locally': False}}
    return trajectory


def _coding(ref, limit=None):
    from . import agents
    for receipt, rid, row in acquired_rows(ref, limit):
        try:
            meta = row.get('metadata') or {}
            teacher_meta = meta.get('teacher_model') if isinstance(meta, dict) else None
            teacher = teacher_meta.get('name') if isinstance(teacher_meta, dict) else teacher_meta
            tools = [agents._prune(json.loads(value) if isinstance(value, str) else value)
                     for value in row.get('tools') or []]
            trajectory = _reported_session(ref, receipt, rid, row,
                row.get('messages') if ref == 'D22' else row.get('trajectory'),
                task_id=row.get('instance_id'), session_id=row.get('trajectory_id'),
                teacher=teacher, tools=tools)
            yield rid, trajectory
        except (Reject, ValueError, TypeError, KeyError) as error:
            yield rid, Reject(str(error))


def open_swe_acquired(limit=None):
    yield from _coding('D22', limit)


def swe_rebench_acquired(limit=None):
    yield from _coding('D25', limit)


def nemotron_terminal_acquired(limit=None):
    for receipt, rid, row in acquired_rows('D35', limit):
        try:
            trajectory = _reported_session('D35', receipt, rid, row, row.get('conversations'),
                task_id=row.get('task'), session_id=rid, teacher=row.get('model'))
            trajectory.lineage['notes']['original_terminal'] = {key: row.get(key) for key in
                ('agent', 'model', 'model_provider', 'date', 'task', 'episode', 'run_id',
                 'trial_name', 'enable_thinking', 'source')}
            yield rid, trajectory
        except (Reject, ValueError, TypeError, KeyError) as error:
            yield rid, Reject(str(error))


def cooperdata_acquired(limit=None):
    for receipt, rid, row in acquired_rows('D17', limit):
        trajectories = row.get('trajectories') or []
        if not trajectories:
            yield rid, Reject('missing original agent trajectories; communication log retained in pinned raw source')
            continue
        for index, agent in enumerate(trajectories):
            child = f'{rid}:agent:{index}'
            try:
                raw = agent.get('messages_json')
                messages = json.loads(raw) if isinstance(raw, str) else raw
                if isinstance(messages, dict):
                    messages = messages.get('messages') or messages.get('trajectory')
                trajectory = _reported_session('D17', receipt, child, row, messages,
                    task_id=row.get('pair_id'), session_id=f'{rid}:{agent.get("agent") or index}',
                    teacher=row.get('model'))
                trajectory.lineage['notes']['original_cooperation'] = {key: row.get(key) for key in
                    ('source', 'source_type', 'pair_id', 'repo', 'task_id', 'features', 'fa_fb',
                     'setting', 'bucket', 'split', 'bucket_reason', 'coord_channel', 'both_passed',
                     'score', 'session_success', 'merge_status', 'conversation', 'agents')}
                trajectory.split_groups.append(group_key('cooper-task', f'{row.get("repo")}:{row.get("task_id")}'))
                if row.get('both_passed') is False:
                    trajectory.outcome['label'] = 'failed'
                trajectory.outcome['details']['reported_both_passed'] = row.get('both_passed')
                yield child, trajectory
            except (Reject, ValueError, TypeError, KeyError) as error:
                yield child, Reject(str(error))


def general_agent_acquired(ref, limit=None):
    """Structured upstream sessions; no tool observations or rewards invented."""
    from . import agents
    for receipt, rid, row in acquired_rows(ref, limit):
        try:
            if not isinstance(row, dict):
                raise Reject('source data row is not an object')
            path = rid.rsplit(':', 1)[0]
            if ref == 'D36' and not path.startswith('trajectories/'):
                raise Reject('duplicate tokenized representation; original trajectories are the consumer')
            if row.get('images'):
                raise Reject('image-bearing session requires supported multimodal input; raw retained')
            raw = row.get('messages') or row.get('conversations')
            if isinstance(raw, str):
                raw = json.loads(raw)
            tools = row.get('tools') or row.get('available_tools')
            if isinstance(tools, str):
                tools = json.loads(tools)
            metadata = row.get('metadata') or {}
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            bound_row = dict(row)
            for key in ('instance_id', 'repo', 'resolved', 'reward', 'status'):
                if bound_row.get(key) is None and key in metadata:
                    bound_row[key] = metadata[key]
            task = bound_row.get('instance_id') or row.get('question') or row.get('uuid') or row.get('trajectory_id') or rid
            teacher = row.get('model') or row.get('teacher_model') or metadata.get('model')
            trajectory = _reported_session(ref, receipt, rid, bound_row, raw,
                task_id=task, session_id=row.get('uuid') or row.get('trajectory_id') or rid,
                teacher=teacher, tools=tools)
            trajectory.lineage['notes']['original_source_metadata'] = {key: row.get(key) for key in
                ('subset_name', 'source', 'domain', 'exit_status', 'trajectory_format',
                 'reward', 'status', 'used_in', 'question_quality_assessment', 'response_quality_assessment')}
            if bound_row.get('reward') is not None and float(bound_row['reward']) <= 0:
                trajectory.outcome['label'] = 'failed'
            if row.get('exit_status') in ('incomplete', 'error', 'failed') or bound_row.get('status') in ('error', 'failed'):
                trajectory.outcome['label'] = 'failed'
            trajectory.outcome['details'].update(reported_reward=bound_row.get('reward'),
                reported_exit_status=row.get('exit_status'), reported_status=bound_row.get('status'))
            yield rid, trajectory
        except (Reject, ValueError, TypeError, KeyError) as error:
            yield rid, Reject(str(error))


def toucan_acquired(limit=None):
    yield from general_agent_acquired('D26', limit)


def toolmind_acquired(limit=None):
    yield from general_agent_acquired('D27', limit)


def nemotron_agentic_acquired(limit=None):
    yield from general_agent_acquired('D28', limit)


def ultradata_acquired(limit=None):
    yield from general_agent_acquired('D38', limit)


def swe_zero_acquired(limit=None):
    yield from general_agent_acquired('D23', limit)


def coderforge_acquired(limit=None):
    yield from general_agent_acquired('D36', limit)


def orchard_acquired(limit=None):
    yield from general_agent_acquired('D37', limit)


def _cooperation_neighbors(ref,receipt,path,known):
    from . import agents
    root=agents.EXPANSION_ROOT/ref
    values={};bindings={};parent=Path(path).parent
    for name in ('eval.json','result.json','metadata.json','conversation.json'):
        relative=str(parent/name);entry=known.get(relative)
        if entry is None:continue
        source=_bound_acquired_path(root,entry)
        # These files are original metadata/communication documents, not a
        # collection whose list elements should be interpreted as sessions.
        with source.open(encoding='utf-8') as stream:values[name]=json.load(stream)
        bindings[name]=dict(entry)
    return values,bindings


def _cooperation_acquired(ref,limit=None,*,rows=None):
    """One complete-context mechanism for original and wrapped Cooper sources."""
    from . import agents
    known={};previous=None
    for receipt,rid,row in (acquired_rows(ref,limit) if rows is None else rows):
        if receipt is not previous:
            known={entry['path']:entry for entry in receipt.get('files',[])};previous=receipt
        path=row.get('_cooperation_member_path',rid.rsplit(':',1)[0]) if isinstance(row,dict) else rid.rsplit(':',1)[0]
        if isinstance(row,dict) and row.get('_archive_omission'):
            yield rid,Reject(row['_archive_omission']);continue
        if not (ref=='D12' and isinstance(row,dict) and row.get('_cooperation_archive_selected') or cooperation_trajectory_file(ref,path,known)):
            yield rid,Reject('non-session or duplicate summary; complete original agent representation is the consumer')
            continue
        try:
            if not isinstance(row,dict):raise Reject('cooperation source is not an original trajectory object')
            if ref=='D12':neighbors,neighbor_bindings=row['_cooperation_neighbors'],row['_cooperation_neighbor_bindings']
            else:neighbors,neighbor_bindings=_cooperation_neighbors(ref,receipt,path,known)
            parts=Path(path).parent.parts
            repo=row.get('repo') or (parts[-3] if len(parts)>=3 else None)
            task_id=row.get('task_id')
            if task_id is None:task_id=parts[-2] if len(parts)>=2 else None
            if repo is None or task_id is None:raise Reject('cooperation source lacks exact original benchmark task')
            for value in neighbors.values():
                if isinstance(value,dict) and (value.get('repo') not in (None,repo)
                        or value.get('task_id') is not None and str(value['task_id'])!=str(task_id)):
                    raise Reject('bound cooperation neighbor belongs to another original benchmark task')
            info=row.get('info') or row.get('mini_swe_info') or {}
            if not isinstance(info,dict):raise Reject('cooperation source metadata is not an object')
            config=info.get('config') or {}
            if not isinstance(config,dict) or not isinstance(config.get('model') or {},dict):
                raise Reject('cooperation source model configuration is not an object')
            teacher=row.get('model') or (config.get('model') or {}).get('model_name')
            task=str(Path(path).parent)
            segments=row.get('segments') if row.get('segments') is not None else row.get('mini_swe_segments') or []
            if not isinstance(segments,list):raise Reject('cooperation source segment inventory is not an original list')
            contexts=[(f'segment:{index}',segment.get('kind') if isinstance(segment,dict) else None,
                       segment.get('messages') if isinstance(segment,dict) else None) for index,segment in enumerate(segments)]
            final=row.get('messages')
            duplicate=next((name for name,_,messages in contexts if messages==final and isinstance(final,list)),None)
            if duplicate is None:contexts.append(('final','final',final))
            source_format=row.get('trajectory_format') or row.get('mini_swe_trajectory_format')
            for name,kind,messages in contexts:
                child=rid+':'+name
                try:
                    if not isinstance(messages,list) or any(not isinstance(item,dict) for item in messages):
                        raise Reject('missing original cooperation context messages')
                    terminal=[{'index':index,'event':item} for index,item in enumerate(messages) if item.get('role')=='exit']
                    context=[];observations=[]
                    for index,item in enumerate(messages):
                        if item.get('role')=='exit':continue
                        if item.get('role') not in ('system','developer','user','assistant','tool','function'):
                            raise Reject('unsupported original cooperation role; context retained in pinned source')
                        if (source_format=='codex_stream_log' and item.get('role')=='assistant'
                                and isinstance(item.get('content'),str) and item['content'].startswith('[command] ')):
                            observations.append({'index':index,'original_role':'assistant','kind':'combined-command-and-observation'})
                            item={**item,'role':'tool'}
                        context.append(item)
                    t=_reported_session(ref,receipt,child,row,context,task_id=task,session_id=child,teacher=teacher)
                    if ref=='D12' and row.get('_cooperation_archive_context',{}).get('view')=='complete-observed-stream-to-exact-final-response':
                        t.messages=context
                    status=row.get('status') or info.get('exit_status')
                    evaluation=neighbors.get('eval.json') or {}
                    if not isinstance(evaluation,dict):raise Reject('cooperation evaluation facts are not an original object')
                    failed=bool(status and status not in ('Submitted','Completed','success'))
                    failed=failed or any(evaluation.get(key) is False for key in ('correct','both_passed')) or evaluation.get('eval')=='fail'
                    failed=failed or any(isinstance(evaluation.get(key),dict) and evaluation[key].get('passed') is False
                                         for key in ('feature1','feature2'))
                    if failed:t.outcome['label']='failed'
                    t.outcome['details'].update(reported_exit_status=status,reported_evaluation=evaluation,
                        reported_evaluation_present='eval.json' in neighbors,executed_locally=False)
                    t.lineage['notes']['original_cooperation_context']={'path':path,'context':name,'kind':kind,
                        'info':info,'trajectory_format':source_format,'repo':repo,'task_id':task_id,
                        'agent_id':row.get('agent_id'),'feature_id':row.get('feature_id'),'team':row.get('team'),
                        'neighbors':neighbors,'neighbor_bindings':neighbor_bindings,
                        'final_duplicate_of':duplicate,'original_segment_count':len(segments),
                        'original_message_count':len(messages),'model_message_count':len(context),
                        'terminal_events':terminal,'observed_command_logs':observations,
                        'archive_context':row.get('_cooperation_archive_context')}
                    if row.get('_natlang_source_projection') is not None:
                        t.lineage['notes']['source_projection'] = row['_natlang_source_projection']
                    t.split_groups.extend([group_key('cooper-task',task),group_key('cooper-task',f'{repo}:{task_id}')])
                    if f'{repo}:{task_id}' in agents._bench_ids():t.split='test'
                    yield child,t
                except (Reject,ValueError,TypeError,KeyError) as error:yield child,Reject(str(error))
        except (Reject,ValueError,TypeError,KeyError) as error:yield rid,Reject(str(error))


def qwen_cooperation_acquired(limit=None):
    yield from _cooperation_acquired('D14',limit)


def team_cooperation_acquired(limit=None):
    yield from _cooperation_acquired('D13',limit)


def openresearcher_structured_acquired(limit=None):
    """Retain research channels and actual recipient dispatch/observation order."""
    from . import agents
    from .trajectory import tool_call
    for receipt,rid,row in acquired_rows('D32',limit):
        try:
            if not isinstance(row,dict) or not isinstance(row.get('messages'),list):
                raise Reject('structured research session has no original messages')
            messages,tools=agents.harmony_messages(row['messages'])
            metadata=[{key:item.get(key) for key in ('role','channel','recipient','content_type','name')}
                      for item in row['messages']]
            task=row.get('qid')
            if task is None:raise Reject('research session lacks original question identity')
            session=f'{rid}:chunk:{row.get("chunk_idx")}'
            t=_reported_session('D32',receipt,rid,row,messages,task_id=str(task),session_id=session,
                teacher=row.get('model') or row.get('teacher_model'),tools=tools)
            t.family='trajectory_continuation_research';t.noun='research session'
            t.lineage['notes']['original_research_channels']={'messages':metadata,'question':row.get('question'),
                'answer':row.get('answer'),'chunk_idx':row.get('chunk_idx'),'num_chunks':row.get('num_chunks'),
                'status':row.get('status'),'error':row.get('error')}
            if row.get('status') not in (None,'success','completed') or row.get('error'):
                t.outcome['label']='failed'
            t.outcome['details'].update(reported_status=row.get('status'),reported_error=row.get('error'))
            yield rid,t
        except (Reject,ValueError,TypeError,KeyError) as error:yield rid,Reject(str(error))


def _collusion_messages(episode,actor,agent):
    """Restore reasoning only through accepted call/shape/ordinal evidence.

    Raw schema1.0.0 records usage rounds/attempts one-based and reasoning trace
    rounds/attempts zero-based. The original full context and exact current task
    prompt delimit this episode; rejected API responses are never attached.
    """
    import copy
    messages=copy.deepcopy(agent.get('messages'))
    if not isinstance(messages,list) or not messages or any(not isinstance(item,dict) for item in messages):
        raise Reject('raw collusion agent has no original messages')
    traces=[(i,t) for i,t in enumerate(episode.get('llm_reasoning_traces') or []) if t.get('agent_id')==actor]
    alignment=[]
    try:
        starts=[i for i,m in enumerate(messages) if m.get('role')=='user' and m.get('content')==agent.get('own_task_prompt')]
        if len(starts)!=1:raise Reject('reasoning lacks one exact original current-episode task boundary')
        positions=[i for i,m in enumerate(messages) if i>starts[0] and m.get('role')=='assistant']
        usage=[(i,u) for i,u in enumerate(episode.get('llm_usage') or [])
               if u.get('actor')==actor and u.get('response_status')=='accepted']
        if len(positions)!=len(usage):raise Reject('accepted usage does not cover original current-episode assistant inventory')
        bindings={}
        for position,(usage_index,u) in zip(positions,usage):
            m=messages[position]
            if (u.get('visible_content_chars')!=len(m.get('content') or '')
                    or u.get('tool_call_count')!=len(m.get('tool_calls') or [])):
                raise Reject('accepted usage differs from exact original assistant response shape')
            values=[]
            for key in ('round','agent_attempt'):
                value=u.get(key)
                if value is not None and (type(value) is not int or value<1):raise Reject('invalid one-based raw call coordinate')
                values.append(None if value is None else value-1)
            key=(u.get('phase'),*values)
            if key in bindings:raise Reject('ambiguous accepted raw call identity')
            bindings[key]=(position,usage_index)
        seen=set()
        for trace_index,trace in traces:
            key=(trace.get('phase'),trace.get('round'),trace.get('attempt'))
            if key in seen or key not in bindings:raise Reject('reasoning trace does not uniquely bind an accepted original call')
            seen.add(key);position,usage_index=bindings[key]
            raw=trace.get('reasoning_trace')
            values=list(raw.values()) if isinstance(raw,dict) else [raw]
            text=[]
            for value in values:
                if value is None or value=='':continue
                value=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False)
                if value not in text:text.append(value)
            reasoning='\n'.join(text)
            if messages[position].get('reasoning_content') not in (None,'',reasoning):
                raise Reject('bound reasoning conflicts with original assistant reasoning')
            if reasoning:messages[position]['reasoning_content']=reasoning
            alignment.append({'trace_index':trace_index,'usage_index':usage_index,'message_index':position,
                              'coordinate_policy':'raw-schema-1.0.0-one-based-usage/zero-based-trace'})
        return messages,{'complete':True,'bindings':alignment}
    except Reject as error:
        # The source session is still usable; separate reasoning cannot be
        # silently attached to a guessed target. Retain every trace for repair.
        return copy.deepcopy(agent['messages']),{'complete':False,'reason':str(error),'bindings':[]}


def agent_collusion_raw_acquired(limit=None):
    """Original full per-agent histories; Parquet reordered summaries are duplicates."""
    for receipt,rid,row in acquired_rows('D19',limit):
        path=rid.rsplit(':',1)[0]
        if not path.startswith('traces/') or not path.endswith('.json.gz'):
            yield rid,Reject('duplicate summary or non-session; original complete raw traces are the consumer')
            continue
        try:
            if not isinstance(row,dict) or not isinstance(row.get('results'),list):
                raise Reject('raw collusion trace has no original episode inventory')
            run=row.get('run_id')
            if not run:raise Reject('raw collusion trace lacks exact original run identity')
            config=row.get('run_config') or {}
            task_groups=sorted({str(task) for episode in row['results'] if isinstance(episode,dict) for task in episode.get('task_ids') or []})
            for episode_index,episode in enumerate(row['results']):
                if not isinstance(episode,dict):
                    yield rid+f':episode:{episode_index}',Reject('raw episode is not an original object')
                    continue
                agents=episode.get('agents')
                if not isinstance(agents,dict):
                    yield rid+f':episode:{episode_index}',Reject('raw episode lacks its actual agent sessions')
                    continue
                for actor,agent in agents.items():
                    child=rid+f':episode:{episode_index}:agent:{actor}'
                    try:
                        if not isinstance(agent,dict):raise Reject('raw episode agent is not an original session object')
                        if row.get('schema_version')=='1.0.0':messages,alignment=_collusion_messages(episode,actor,agent)
                        else:
                            messages=agent.get('messages');alignment={'complete':False,'reason':'unqualified raw reasoning coordinate schema','bindings':[]}
                        t=_reported_session('D19',receipt,child,{},messages,task_id=run,session_id=child,
                            teacher=(config.get('models') or {}).get(actor))
                        t.split_groups.extend(group_key('agent-collusion-task',task) for task in task_groups)
                        t.split_groups.append(group_key('agent-collusion-run',run))
                        facts={key:value for key,value in agent.items() if key!='messages'}
                        t.lineage['notes']['original_collusion_session']={'path':path,'run_id':run,
                            'condition_id':row.get('condition_id'),'sequence_id':row.get('sequence_id'),
                            'schema_version':row.get('schema_version'),'run_config':config,
                            'episode_index':episode_index,'episode_id':episode.get('episode_id'),'actor':actor,
                            'agent':facts,'events':episode.get('events'),'channel_transcript':episode.get('channel_transcript'),
                            'analysis_metadata':episode.get('analysis_metadata'),
                            'llm_reasoning_traces':[trace for trace in episode.get('llm_reasoning_traces') or [] if trace.get('agent_id')==actor],
                            'llm_usage':[usage for usage in episode.get('llm_usage') or [] if usage.get('actor')==actor],
                            'reasoning_alignment':alignment}
                        checks=[agent.get(key) for key in ('extraction_eval','data_search_eval','code_eval','code_assessment')]
                        failed=agent.get('verdict_correct') is False
                        failed=failed or any(isinstance(check,dict) and (check.get('quality_verdict')=='fail'
                            or check.get('passed') is False or check.get('correct') is False) for check in checks)
                        if agent.get('reward') is not None and float(agent['reward'])<=0:failed=True
                        if failed:t.outcome['label']='failed'
                        t.outcome['details'].update(reported_reward=agent.get('reward'),
                            reported_verdict_correct=agent.get('verdict_correct'),reported_true_expected_verdict=agent.get('true_expected_verdict'),
                            reported_task_checks=checks,reasoning_alignment_complete=alignment['complete'])
                        yield child,t
                    except (Reject,ValueError,TypeError,KeyError) as error:yield child,Reject(str(error))
        except (Reject,ValueError,TypeError,KeyError) as error:yield rid,Reject(str(error))
