"""Exact whole-graph deletion and authenticated own-index continuation."""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
import tempfile

from natlang_neuralese.common.hashing import canonical_json_sha256_hex as digest, sha256_file_hex
from natlang_neuralese.common import row_index
from scripts.neuralese_data.continue_native_source_indexes import _publish_index_without_replacement


def _state(path):
    value = Path(path).stat()
    return tuple(getattr(value, key) for key in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns'))


@contextmanager
def _raw(path):
    if str(path).endswith('.zst'):
        child = subprocess.Popen(['zstd','-qdc',str(path)], stdout=subprocess.PIPE)
        try:
            yield child.stdout
        finally:
            failed = sys.exc_info()[0] is not None
            child.stdout.close()
            if child.wait() and not failed:
                raise ValueError('native source decompression failed')
    else:
        with Path(path).open('rb') as stream:
            yield stream


def filter_source(source, output, rows, *, source_sha256):
    """Remove enumerated complete graphs; preserve remaining raw bytes/order."""
    source, output = Path(source), Path(output)
    before = _state(source)
    if sha256_file_hex(source) != source_sha256 or source.resolve() == output.resolve():
        raise ValueError('withdrawal source identity differs')
    selected = {}
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get('id'), str) or row['id'] in selected
                or not isinstance(row.get('binding'), str) or re.fullmatch('[0-9a-f]{64}',row['binding']) is None
                or row.get('source_sha256') != source_sha256
                or type(row.get('communication')) is not bool):
            raise ValueError('withdrawal rows need exact unique source bindings')
        selected[row['id']] = row
    if not selected:
        raise ValueError('withdrawal needs enumerated targets')
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.'+output.name+'.', dir=output.parent)
    temporary = Path(temporary)
    old_ids, kept_ids, removed = [], [], []
    seen = set(); retained = hashlib.sha256()
    try:
        with os.fdopen(descriptor, 'wb') as target:
            compressor = (subprocess.Popen(['zstd','-q','-3','--long=27','-T1','-c'],
                          stdin=subprocess.PIPE, stdout=target) if str(output).endswith('.zst') else None)
            writer = compressor.stdin if compressor else target
            try:
                with _raw(source) as stream:
                    for raw in stream:
                        if not raw.strip():
                            writer.write(raw); retained.update(raw); continue
                        row = json.loads(raw); ident = row['id']
                        if not isinstance(ident, str) or ident in seen:
                            raise ValueError('withdrawal source repeats a graph identity')
                        seen.add(ident); old_ids.append(ident)
                        if ident in selected:
                            from natlang_neuralese.data.native_graph_schema import parse_graph, communication_classification
                            from natlang_neuralese.data.native_sources import _row_bindings
                            fingerprint = row.get('tokenizer_fingerprint')
                            if not isinstance(fingerprint, str) or not fingerprint:
                                raise ValueError('withdrawal target tokenizer identity is missing')
                            example = parse_graph(row, tokenizer_fingerprint=fingerprint)
                            classification = communication_classification(example)['communication']
                            if (example.split != 'train' or classification is not selected[ident]['communication']
                                    or _row_bindings([example])[ident] != selected[ident]['binding']):
                                raise ValueError('withdrawal target graph class or binding differs')
                            removed.append({**selected[ident], 'tokenizer_fingerprint':fingerprint,
                                            'raw_row_sha256':hashlib.sha256(raw).hexdigest()})
                        else:
                            writer.write(raw); retained.update(raw); kept_ids.append(ident)
            finally:
                if compressor:
                    failed = sys.exc_info()[0] is not None
                    try:
                        writer.close()
                    except OSError:
                        if not failed:
                            raise
                    finally:
                        status = compressor.wait()
                    if status and not failed:
                        raise ValueError('filtered native source compression failed')
            target.flush(); os.fsync(target.fileno())
        if {row['id'] for row in removed} != set(selected) or not kept_ids:
            raise ValueError('withdrawal targets missing or source entirely removed')
        check = hashlib.sha256()
        # Temporary names do not carry the compression suffix.
        if str(output).endswith('.zst'):
            child = subprocess.Popen(['zstd','-qdc',str(temporary)], stdout=subprocess.PIPE)
            with child.stdout as stream:
                for chunk in iter(lambda:stream.read(1024*1024), b''):check.update(chunk)
            if child.wait():raise ValueError('filtered native source readback failed')
        else:
            with temporary.open('rb') as stream:
                for chunk in iter(lambda:stream.read(1024*1024), b''):check.update(chunk)
        if check.digest() != retained.digest() or _state(source) != before or sha256_file_hex(source) != source_sha256:
            raise ValueError('withdrawal retained bytes/source changed')
        output_sha = sha256_file_hex(temporary)
        try:os.link(temporary, output)
        except FileExistsError:
            if sha256_file_hex(output) != output_sha:raise ValueError('filtered destination conflicts')
        directory = os.open(output.parent, os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(directory)
        finally:os.close(directory)
        return {'schema':'natlang.native-source-exact-withdrawal-filter/1','source_sha256':source_sha256,
                'output':{'path':str(output),'sha256':output_sha,'bytes':output.stat().st_size},
                'old_rows':len(old_ids),'new_rows':len(kept_ids),'old_ids_sha256':digest(old_ids),
                'new_ids_sha256':digest(kept_ids),'retained_raw_sha256':retained.hexdigest(),
                'removed':removed,'training_admission':False,'applied_to_training':False}
    finally:
        temporary.unlink(missing_ok=True)


def continue_filtered_index(index, proof, *, index_sha256, metadata_sha256, output_path=None):
    """Project an authenticated own index through the exact raw deletion proof."""
    if proof.get('schema') != 'natlang.native-source-exact-withdrawal-filter/1':
        raise ValueError('withdrawal filter proof schema differs')
    index = Path(index); before = _state(index)
    if sha256_file_hex(index) != index_sha256:
        raise ValueError('withdrawal predecessor index changed')
    # Only an explicitly SHA-bound own cache is deserialized.
    with index.open('rb') as stream:value = pickle.load(stream)
    entries = value['entries']; key = value['key']
    if (value.get('schema') != 'natlang.row-index/1' or value['metadata_sha256'] != metadata_sha256
            or digest(entries) != metadata_sha256 or key['source_sha256'] != proof['source_sha256']
            or key['kind'] != 'native-graphs' or key['split'] != 'train'
            or digest([row['id'] for row in entries]) != proof['old_ids_sha256']):
        raise ValueError('withdrawal index/source/metadata differs')
    removed = {row['id']:row for row in proof['removed']}
    if len(removed) != len(proof['removed']):raise ValueError('withdrawal repeats an index target')
    actual = {row['id']:row for row in entries if row['id'] in removed}
    if (set(actual) != set(removed) or any(actual[k]['binding'] != removed[k]['binding']
            or removed[k].get('tokenizer_fingerprint') != key.get('tokenizer')
            or type(removed[k].get('communication')) is not bool
            or type(actual[k].get('communication')) is not bool
            or actual[k]['communication'] is not removed[k]['communication'] for k in removed)):
        raise ValueError('withdrawal index target binding differs')
    kept = [row for row in entries if row['id'] not in removed]
    # Transport keeps the original filter proof intact; bind the actual local copy.
    output = Path(output_path or proof['output']['path']); output_before = _state(output)
    if (len(entries) != proof['old_rows'] or len(kept) != proof['new_rows']
            or digest([row['id'] for row in kept]) != proof['new_ids_sha256']
            or sha256_file_hex(output) != proof['output']['sha256']
            or output.stat().st_size != proof['output']['bytes'] or _state(index) != before):
        raise ValueError('withdrawal retained source/index differs')
    raw_digest = hashlib.sha256()
    with _raw(output) as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b''):
            raw_digest.update(chunk)
    if raw_digest.hexdigest() != proof['retained_raw_sha256'] or _state(output) != output_before:
        raise ValueError('withdrawal retained raw payload differs')
    new_key = {**key,'source_sha256':proof['output']['sha256']}
    target = index.parent / row_index.entry_path('native-graphs',digest(new_key)).name
    result = {'schema':'natlang.row-index/1','key':new_key,'entries':kept,'metadata_sha256':digest(kept)}
    _publish_index_without_replacement(target,result)
    if _state(index) != before or _state(output) != output_before:
        raise ValueError('withdrawal source/index changed while publishing')
    return {'path':str(target),'sha256':sha256_file_hex(target),'metadata_sha256':result['metadata_sha256'],
            'rows':len(kept),'old_index_sha256':index_sha256,'policy_sha256':key['policy_sha256']}
