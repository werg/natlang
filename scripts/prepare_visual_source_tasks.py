#!/usr/bin/env python3
"""Build artifact task packets from pinned public sources; never execute imported code.

This is a source adapter, not a SkillEpisode collector or training admission step.
Host oracles and references are stored separately from model-visible inputs.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import uuid
import xml.etree.ElementTree as ET

from acquire_visual_sources import contained, digest, safe_path, write_json


def sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


class Resources(HTMLParser):
    def __init__(self):
        super().__init__()
        self.resources = []
    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if value and (name in ('src', 'srcset', 'poster') or (tag == 'link' and name == 'href')):
                self.resources.append(value)


def assets_in(code):
    parser = Resources()
    parser.feed(code)
    values = parser.resources + re.findall(r'url\(\s*[\"\']?([^\)\"\']+)', code, re.I)
    return sorted(set(value for value in values if not value.startswith(('#', 'data:'))))


def read_parquet(path):
    import pyarrow.parquet as pq
    table = pq.ParquetFile(path)
    # Bounded batches, never load a whole multi-GB source shard into host RAM.
    for batch in table.iter_batches(batch_size=16):
        yield from batch.to_pylist()


def normalize_value(value, directory):
    if isinstance(value, bytes):
        h = sha(value)
        path = directory / 'assets' / h
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(value)
        return {'asset': path.relative_to(directory).as_posix(), 'sha256': h, 'bytes': len(value)}
    if isinstance(value, dict):
        return {key: normalize_value(item, directory) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_value(item, directory) for item in value]
    return value


def source_rows(source, files):
    kind = source['adapter']
    if kind == 'cadtests':
        for partition in ('abstract', 'detailed'):
            tests = defaultdict(list)
            join_bytes = 0
            for row in read_parquet(files / f'cadtests/{partition}.parquet'):
                if row['partition'] != partition:
                    raise ValueError('CADTest partition differs from file')
                tests[str(row['sample_id'])].append(row)
                if len(tests) > source['limit'] or sum(len(v) for v in tests.values()) > source['limit'] * 100:
                    raise ValueError('CADTest join exceeds declared sample/check caps')
                row_bytes = len(canonical(row).encode())
                join_bytes += row_bytes
                if row_bytes > source['max_artifact_bytes'] or join_bytes > source['max_artifact_bytes'] * 10:
                    raise ValueError('CADTest check row/join exceeds byte cap')
            seen = set()
            for index, row in enumerate(read_parquet(files / f'samples/{partition}.parquet')):
                if index >= source['limit']:
                    raise ValueError('CADTest sample partition exceeds declared limit')
                key = str(row['sample_id'])
                if row['partition'] != partition or key in seen or key not in tests:
                    raise ValueError('CADTest duplicate, missing tests or wrong partition')
                seen.add(key)
                yield f'samples/{partition}.parquet', index, {**row, 'cadtests': tests[key]}
            if seen != set(tests):
                raise ValueError('CADTest tests lack prompt rows')
    elif kind in ('web-html', 'svg'):
        for name in source['files']:
            for index, row in enumerate(read_parquet(files / name)):
                if index >= source['limit']:
                    break
                yield name, index, row
    elif kind in ('designbench', 'designbench-frameworks'):
        for name in source['files']:
            if (files / name).stat().st_size > source['max_artifact_bytes']:
                raise ValueError('DesignBench JSON exceeds artifact cap')
            yield name, None, json.loads((files / name).read_text())
    elif kind == 'html-image':
        for name in source['files']:
            if name.endswith('.html'):
                yield name, None, {'html': (files / name).read_text(), 'image_file': name[:-5] + '.png'}
    elif kind == 'cadgen':
        import yaml
        for name in source['files']:
            if name.endswith('/description.yaml'):
                if (files / name).stat().st_size > source['max_artifact_bytes']:
                    raise ValueError('CAD description exceeds artifact cap')
                row = yaml.safe_load((files / name).read_text())
                if not isinstance(row, dict):
                    raise ValueError('CAD description must be an object')
                yield name, None, row
    else:
        raise ValueError('unimplemented source adapter: ' + kind)


def adapt(source, name, index, row, files):
    kind = source['adapter']
    blockers = ['independent_artifact_executor_pending']
    review = source['review_state']
    if review != 'source_assets_and_replay_pending':
        blockers.append(review)
    oracle = {}
    if kind == 'cadtests':
        prompt = row['prompt']
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('empty CAD task prompt')
        model = {'modality': 'text', 'prompt': prompt, 'output_contract': 'Python CadQuery code defining final_result'}
        blockers.append('cadprompt_upstream_rights_pending')
        oracle = {'checks': row['cadtests'], 'review': 'Requirement-to-prompt agreement, tolerance and mutation audit required before using these checks as rewards.'}
        # Abstract and detailed views of the same CADPrompt program are one group.
        group = 'cadprompt/program/' + str(row['sample_id'])
        family = 'parametric-3d-cad'
    elif kind in ('designbench', 'designbench-frameworks'):
        framework = name.split('/')[1]
        operation = name.split('/')[0]
        if row.get('framework', framework) != framework:
            raise ValueError('framework metadata differs from source path')
        def file_map(code):
            if framework == 'angular':
                if not isinstance(code, dict) or not code or any(k not in ('html', 'ts', 'css') for k in code):
                    raise ValueError('Angular source requires typed html/ts/css file map')
                mapping = {'html': 'new.component.html', 'ts': 'new.component.ts', 'css': 'new.component.css'}
                if not all(isinstance(v, str) and v.strip() for v in code.values()):
                    raise ValueError('Angular file content must be nonempty strings')
                return {mapping[k]: v for k, v in code.items()}
            if not isinstance(code, str) or not code.strip():
                raise ValueError('frontend source code absent')
            return {{'vanilla': 'index.html', 'react': 'App.jsx', 'vue': 'App.vue'}[framework]: code}
        if operation == 'edit':
            prompt, original, reference = row.get('prompt'), row.get('src_code'), row.get('dst_code')
        elif operation == 'compile':
            issue = row.get('issue')
            if not isinstance(issue, str) or not issue.strip():
                raise ValueError('compile repair requires visible error diagnostics')
            prompt = 'Repair the supplied component while preserving its intended UI and behavior. Compiler diagnostics: ' + issue
            original, reference = row.get('code'), row.get('repaired_code')
        else:
            raise ValueError('unsupported frontend operation')
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('visible edit prompt absent')
        original_files, reference_files = file_map(original), file_map(reference)
        model = {'modality': 'text', 'prompt': prompt, 'files': original_files,
            'framework': framework, 'output_contract': 'Frontend artifact file map'}
        oracle = {'reference_files': reference_files,
            'checks': 'Independent requirement/behavior and rendering checks required; upstream compile or aesthetic annotations are not rewards.'}
        grouping_code = row.get('component_jsx') or canonical(original_files)
        group = 'frontend/source/' + sha(re.sub(r'\s+', ' ', grouping_code).strip())
        if framework == 'angular':
            from pathlib import PurePosixPath
            missing = []
            for code in original_files.values():
                for ref in re.findall(r'(?:templateUrl|styleUrl)\s*:\s*[\"\']([^\"\']+)', code):
                    normalized = PurePosixPath(ref).as_posix()
                    if normalized not in original_files:
                        missing.append(ref)
                for style_list in re.findall(r'styleUrls\s*:\s*\[([^\]]*)\]', code):
                    for ref in re.findall(r'[\"\']([^\"\']+)[\"\']', style_list):
                        if PurePosixPath(ref).as_posix() not in original_files:
                            missing.append(ref)
            if missing:
                blockers.append('unresolved_relative_component_files')
                oracle['unresolved_relative_component_files'] = sorted(set(missing))
        if framework != 'vanilla':
            blockers.append('pinned_framework_dependency_environment_pending')
        family = 'frontend-compile-repair' if operation == 'compile' else 'frontend-edit'
    elif kind in ('web-html', 'html-image'):
        code = row.get('text') if kind == 'web-html' else row.get('html')
        if not isinstance(code, str) or '<html' not in code.lower():
            raise ValueError('missing HTML document')
        if len(code.encode()) > source['max_artifact_bytes']:
            raise ValueError('HTML exceeds artifact cap')
        if kind == 'web-html':
            model = {'modality': 'text', 'prompt': 'Improve this page for narrow and wide viewports. Preserve all visible text and intended interactions, avoid horizontal overflow and clipped content, and keep the page self-contained. Return the revised HTML/CSS artifact.', 'files': {'index.html': code}, 'output_contract': 'HTML artifact file map'}
            family = 'responsive-html-css'
        else:
            image = row['image_file']
            if not (files / safe_path(image)).is_file():
                raise ValueError('screenshot absent')
            model = {'modality': 'image', 'prompt': 'Implement this reference webpage as HTML/CSS while preserving visible content and layout.', 'image_source_file': image, 'output_contract': 'HTML artifact file map'}
            blockers.append('image_input_replay_training_path_pending')
            family = 'screenshot-to-frontend'
        external = assets_in(code)
        if external:
            blockers.append('external_assets_pending')
        oracle = {'reference_files': {'index.html': code}, 'unresolved_resources': external, 'checks': 'Rendered visible-content preservation and independent multi-viewport layout violations; raw pixels and upstream aesthetics are not sole rewards.'}
        group = 'frontend/source/' + sha(re.sub(r'\s+', ' ', code).strip())
    elif kind == 'svg':
        code = row.get('Svg')
        if not isinstance(code, str) or len(code.encode()) > source['max_artifact_bytes']:
            raise ValueError('missing/oversized SVG')
        if '<!DOCTYPE' in code.upper() or '<!ENTITY' in code.upper():
            raise ValueError('SVG entity declarations unsupported')
        tree = ET.fromstring(code)
        if tree.tag.split('}')[-1] != 'svg':
            raise ValueError('SVG root required')
        prohibited = [element.tag.split('}')[-1] for element in tree.iter() if element.tag.split('}')[-1] in ('script', 'foreignObject', 'image')]
        if prohibited:
            blockers.append('non_vector_or_active_svg_content')
        if any(value and not value.startswith('#') for element in tree.iter() for key, value in element.attrib.items() if key.split('}')[-1] == 'href'):
            blockers.append('external_svg_resources_pending')
        model = {'modality': 'text', 'prompt': 'Refactor this SVG into a more compact editable vector representation while preserving all rendered shapes, text, colors and layering at multiple display sizes. Return SVG code; do not replace vectors with a raster image.', 'files': {'drawing.svg': code}, 'output_contract': 'SVG artifact file map'}
        oracle = {'reference_files': {'drawing.svg': code}, 'checks': 'Independent multiscale rendering/feature equivalence before serialized UTF-8 bytes; no XML-string equality reward.'}
        group = 'svg/source/' + sha(ET.tostring(tree))
        family = 'svg-repair-and-compaction'
    elif kind == 'cadgen':
        prompt = row.get('description')
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('CADGen description absent')
        fixture = name.rsplit('/', 1)[0]
        declared = row.get('input_files', [])
        if not isinstance(declared, list):
            raise ValueError('CADGen input_files must be a list')
        attachments = []
        for item in declared:
            path = fixture + '/' + str(safe_path(item))
            if not (files / path).is_file():
                raise ValueError('CADGen input asset absent: ' + path)
            attachments.append(path)
        # Text edit directions can be in a separate public input file.
        edit_path = files / fixture / 'edit_description.txt'
        if edit_path.exists():
            if edit_path.stat().st_size > source['max_artifact_bytes']:
                raise ValueError('CAD edit description exceeds artifact cap')
            prompt += '\n\n' + edit_path.read_text()
        model = {'modality': row.get('input_type', 'text+image'), 'prompt': prompt, 'source_files': attachments, 'output_contract': 'output.step'}
        blockers.extend(['image_or_step_input_path_pending', 'private_quality_oracle_unavailable'])
        oracle = {'checks': 'Public solid validity can be checked; shape/interface/topology target metrics require unavailable private ground truth.', 'ground_truth_available': False}
        group = 'cadgenbench/fixture/' + fixture
        family = 'parametric-3d-cad'
    else:
        raise ValueError('unknown adapter')
    return family, group, model, oracle, sorted(set(blockers))


def prepare(source, raw_root, out):
    safe_path(source['id'])
    if '/' in source['id']:
        raise ValueError('source ID must be one path component')
    raw = contained(raw_root, raw_root / source['id'] / source['revision'])
    receipt_path = contained(raw, raw / 'acquisition.json')
    receipt = json.loads(receipt_path.read_text())
    spec_sha = sha(canonical(source))
    if receipt.get('source_spec_sha256') != spec_sha or receipt.get('status') != 'complete':
        raise ValueError('acquisition incomplete or source lock changed')
    if receipt.get('executed_upstream_code') is not False:
        raise ValueError('unexpected acquisition execution policy')
    hashes = {}
    for item in receipt['files']:
        path = contained(raw, raw / 'files' / safe_path(item['path']))
        if not path.is_file() or digest(path) != item['sha256'] or path.stat().st_size != item['bytes']:
            raise ValueError('raw source bytes differ: ' + item['path'])
        hashes[item['path']] = item['sha256']
    final_directory = contained(out, out / source['id'])
    if final_directory.exists():
        raise ValueError('output exists; use a fresh version instead of changing prepared tasks')
    directory = contained(out, out / ('.' + source['id'] + '.' + uuid.uuid4().hex + '.partial'))
    directory.mkdir(parents=True)
    tasks, held = [], []
    seen_ids = set()
    for name, index, raw_row in source_rows(source, raw / 'files'):
        location = name + (f'#row={index}' if index is not None else '')
        try:
            family, group, model, oracle, blockers = adapt(source, name, index, raw_row, raw / 'files')
            row = normalize_value(raw_row, directory)
            for label, value in [('source record', row), ('visible input', model), ('host oracle', oracle)]:
                if len(canonical(value).encode()) > source['max_artifact_bytes']:
                    raise ValueError(label + ' exceeds serialized artifact cap')
            ident = source['id'] + '/' + sha(location)[:20]
            if ident in seen_ids:
                raise ValueError('duplicate source identity')
            seen_ids.add(ident)
            record_path = directory / 'records' / (sha(location) + '.json')
            write_json(record_path, row)
            oracle_path = directory / 'oracles' / (sha(location) + '.json')
            write_json(oracle_path, oracle)
            # Explicit separation: the model-visible input never contains oracle fields.
            tasks.append({'schema': 'natlang.artifact-source-task/1', 'id': ident,
                'family': family, 'group': group, 'partition_assignment': 'unassigned',
                'split_review': 'Global cross-source grouping is required before collection; preserve original benchmark use in provenance.',
                'dedup_keys': [sha(canonical(model.get('files', {})))] if model.get('files') else [],
                'source': {'id': source['id'], 'url': source['upstream_url'], 'revision': source['revision'], 'file': name, 'file_sha256': hashes[name], 'row': index, 'record_path': record_path.relative_to(directory).as_posix(), 'record_sha256': digest(record_path), 'dataset_license': source['expected_dataset_license'], 'original_split': 'evaluation' if source['adapter'] in ('designbench', 'designbench-frameworks', 'html-image', 'cadtests', 'cadgen') else 'train', 'benchmark_repurposing_review_required': source['adapter'] in ('designbench', 'designbench-frameworks', 'html-image', 'cadtests', 'cadgen')},
                'input': model, 'host_oracle': {'path': oracle_path.relative_to(directory).as_posix(), 'sha256': digest(oracle_path)},
                'source_assets_root': (raw / 'files').as_posix(),
                'blockers': blockers, 'status': 'prepared_requires_artifact_executor' if blockers == ['independent_artifact_executor_pending'] else 'held_source_or_evaluator_review',
                'training_admitted': False, 'provider_calls': 0})
        except (ValueError, TypeError, KeyError, ET.ParseError) as error:
            held.append({'location': location, 'file_sha256': hashes.get(name), 'reason': str(error)})
    body = ''.join(canonical(task) + '\n' for task in tasks)
    (directory / 'artifact-source-tasks.jsonl').write_text(body)
    write_json(directory / 'held-rows.json', held)
    manifest = {'schema': 'natlang.artifact-source-corpus/1', 'source': source, 'acquisition_receipt_sha256': digest(receipt_path),
        'adapter_artifact': 'adapter.py', 'adapter_sha256': digest(Path(__file__)),
        'helper_artifact': 'acquire_visual_sources.py', 'helper_sha256': digest(Path(__file__).with_name('acquire_visual_sources.py')), 'sha256': sha(body), 'tasks': len(tasks), 'groups': len({t['group'] for t in tasks}),
        'statuses': dict(Counter(t['status'] for t in tasks)), 'blockers': dict(Counter(b for t in tasks for b in t['blockers'])),
        'held_unsupported_rows': len(held), 'held_rows_sha256': digest(directory / 'held-rows.json'), 'provider_calls': 0, 'executed_imported_code': False, 'training_admitted': False}
    (directory / 'adapter.py').write_bytes(Path(__file__).read_bytes())
    (directory / 'acquire_visual_sources.py').write_bytes(Path(__file__).with_name('acquire_visual_sources.py').read_bytes())
    write_json(directory / 'manifest.json', manifest)
    directory.rename(final_directory)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, default=Path('training/visual_sources.json'))
    parser.add_argument('--raw', type=Path, default=Path('vendor/datasets/visual-frontend'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', default='all')
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text())
    selected = set(args.source.split(','))
    sources = [row for row in registry['sources'] if args.source == 'all' or row['id'] in selected]
    if args.source != 'all' and selected != {row['id'] for row in sources}:
        parser.error('unknown source ID')
    for source in sources:
        print(canonical(prepare(source, args.raw, args.out)), flush=True)


if __name__ == '__main__':
    main()
