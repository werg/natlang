"""Evaluate explicitly selected held windows against shared crisp/AR controls.

This command reconstructs inference weights from an exported heads checkpoint;
it does not load a training optimizer or implement a second rollout.  The
selection manifest binds every source row, split, target span, context prefix,
content class, and unavailable stratum.  All scoring delegates to
``projected_history.autoregressive_history_metrics``.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import torch

from ..train.output_embedding_projection import sha
from ..train.text_warmup import ROLE_CODES, chat_roles
from .projected_history import autoregressive_history_metrics


SOURCE_SCHEMA = 'natlang.matched-ar-evaluator-source/1'
SELECTION_SCHEMA = 'natlang.matched-ar-window-selection/1'
WINDOW_KINDS = frozenset({'prose', 'code', 'tool_argument'})


def _subtract_spans(start, end, exclusions):
    result = []
    cursor = start
    for left, right in sorted((a, b) for a, b in exclusions if a < end and b > start):
        left, right = max(start, left), min(end, right)
        if cursor < left:
            result.append((cursor, left))
        cursor = max(cursor, right)
    if cursor < end:
        result.append((cursor, end))
    return result


def _source_class_regions(text):
    """Find exact serialized source regions for prose, code and tool arguments."""
    assistant = []
    for match in re.finditer(r'<\|im_start\|>assistant[^\n]*\n', text):
        end = text.find('<|im_end|>', match.end())
        if end >= 0:
            assistant.append((match.end(), end))
    tool_calls, tool_wrappers = [], []
    marker_open, marker_close = '<|tool_call_start|>', '<|tool_call_end|>'
    cursor = 0
    while True:
        start = text.find(marker_open, cursor)
        if start < 0:
            break
        end = text.find(marker_close, start + len(marker_open))
        if end < 0:
            break
        region = (start + len(marker_open), end)
        if any(left <= region[0] and region[1] <= right for left, right in assistant):
            tool_calls.append(region)
            tool_wrappers.append((start, end + len(marker_close)))
        cursor = end + len(marker_close)
    code = []
    for left, right in assistant:
        body = text[left:right]
        for match in re.finditer(r'```.*?```', body, re.S):
            code.append((left + match.start() + 3, left + match.end() - 3))
    # The code string is a real eval argument in this corpus. Its serialized
    # escaped bytes are the supervised target surface, distinct from its wrapper.
    for left, right in tool_calls:
        payload = text[left:right]
        for match in re.finditer(r"\bcode\s*=\s*'((?:\\.|[^'\\])*)'", payload):
            code.append((left + match.start(1), left + match.end(1)))
    tool_arguments = [part for left, right in tool_calls for part in _subtract_spans(left, right, code)]
    prose_exclusions = tool_wrappers + [span for span in code
                                     if any(left <= span[0] and span[1] <= right
                                            for left, right in assistant)]
    prose = [part for left, right in assistant for part in _subtract_spans(left, right, prose_exclusions)]
    return {'prose': prose, 'code': code, 'tool_argument': tool_arguments}


def _pin_file(pin, label):
    if (not isinstance(pin, dict) or not isinstance(pin.get('path'), str) or
            not isinstance(pin.get('sha256'), str) or len(pin['sha256']) != 64 or
            any(char not in '0123456789abcdef' for char in pin['sha256'])):
        raise ValueError(f'{label} needs a path and SHA-256')
    path = Path(pin['path'])
    if not path.is_file() or sha(path) != pin['sha256']:
        raise ValueError(f'{label} pin does not match its file')
    return path


def _verify_source_recipe(recipe, recipe_path):
    if not isinstance(recipe, dict):
        raise ValueError('source recipe must be an object')
    if recipe.get('schema') != SOURCE_SCHEMA:
        raise ValueError(f'source recipe must use {SOURCE_SCHEMA}')
    required = ('training_recipe', 'source_checkpoint', 'heads_checkpoint', 'text_data')
    pins = {name: _pin_file(recipe.get(name), name) for name in required}
    pairing = recipe.get('source_pairing')
    if not isinstance(pairing, dict) or type(pairing.get('step')) is not int:
        raise ValueError('source_pairing must bind the frozen full-state/heads pair')
    freeze_path = _pin_file(pairing.get('freeze_receipt'), 'source freeze receipt')
    best_manifest_path = _pin_file(pairing.get('frozen_best_manifest'), 'frozen best manifest')
    freeze_receipt = json.loads(freeze_path.read_text())
    best_manifest = json.loads(best_manifest_path.read_text())
    if (freeze_receipt.get('schema') != 'natlang.ar-feedback-frozen-source-pair/1' or
            freeze_receipt.get('status') != 'frozen pair hash/size/inode and matching manifest verified' or
            freeze_receipt.get('source_best_step') != pairing['step'] or
            best_manifest.get('step') != pairing['step']):
        raise ValueError('source checkpoint pairing receipt/manifest is inconsistent')
    frozen_files = freeze_receipt.get('files')
    if not isinstance(frozen_files, dict):
        raise ValueError('source freeze receipt lacks its file bindings')
    for name, pin_name in (('best-checkpoint.pt', 'source_checkpoint'), ('best-heads.pt', 'heads_checkpoint')):
        record = frozen_files.get(name)
        if not isinstance(record, dict) or record.get('sha256') != recipe[pin_name]['sha256']:
            raise ValueError(f'{name} does not match the paired source recipe pin')
        if Path(record.get('path', '')).resolve() != pins[pin_name].resolve():
            raise ValueError(f'{name} path does not match the frozen source recipe pin')
        if pins[pin_name].stat().st_ino != record.get('inode'):
            raise ValueError(f'{name} is no longer the frozen source-pair inode')
    base = recipe.get('base_model')
    if not isinstance(base, dict) or not isinstance(base.get('path'), str) or not isinstance(base.get('files'), dict) or not base['files']:
        raise ValueError('base_model needs an exact local path and a relative file SHA-256 map')
    base_path = Path(base['path'])
    if not base_path.is_dir():
        raise ValueError('base_model path must be a local snapshot directory')
    for relative, expected in base['files'].items():
        if not isinstance(relative, str) or Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError('base_model file paths must be relative and stay inside the snapshot')
        actual = base_path / relative
        if not actual.is_file() or not isinstance(expected, str) or sha(actual) != expected:
            raise ValueError(f'base_model file pin mismatch: {relative}')
    code_snapshot = recipe.get('evaluator_code_snapshot')
    if (not isinstance(code_snapshot, dict) or not isinstance(code_snapshot.get('git_commit'), str) or
            not isinstance(code_snapshot.get('code_root'), str) or
            not isinstance(code_snapshot.get('files'), dict) or not code_snapshot['files']):
        raise ValueError('evaluator_code_snapshot needs a repository root, revision, and exact source-file hashes')
    code_root = Path(code_snapshot['code_root']).resolve()
    if not code_root.is_dir():
        raise ValueError('evaluator code snapshot root is not available')
    code_paths = {}
    for relative, expected in code_snapshot['files'].items():
        if not isinstance(relative, str) or Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError('evaluator code snapshot paths must be repository-relative')
        actual = code_root / relative
        if not actual.is_file() or sha(actual) != expected:
            raise ValueError(f'evaluator code snapshot pin mismatch: {relative}')
        code_paths[relative] = actual.resolve()
    active_metric_module = sys.modules.get(autoregressive_history_metrics.__module__)
    if active_metric_module is None or not isinstance(getattr(active_metric_module, '__file__', None), str):
        raise ValueError('shared AR metrics module has no verifiable source file')
    active_files = (Path(__file__).resolve(), Path(active_metric_module.__file__).resolve())
    for active in active_files:
        try:
            relative = active.relative_to(code_root).as_posix()
        except ValueError as exc:
            raise ValueError('active evaluator code is outside the pinned code snapshot') from exc
        if code_paths.get(relative) != active:
            raise ValueError(f'active evaluator code is not listed in the pinned code snapshot: {relative}')
    context_limit = recipe.get('max_context_tokens')
    if type(context_limit) is not int or context_limit < 2:
        raise ValueError('max_context_tokens must be a positive integer')
    return {
        'pins': pins,
        'base_path': base_path,
        'context_limit': context_limit,
        'recipe_path': Path(recipe_path),
        'recipe_sha256': sha(recipe_path),
        'code_snapshot': code_snapshot,
    }


def token_ids_sha256(ids):
    """Hash integer token IDs as canonical compact JSON, independent of tensor dtype."""
    if not isinstance(ids, (list, tuple)) or any(type(token) is not int or token < 0 for token in ids):
        raise ValueError('token ID sequence must contain nonnegative integers')
    payload = json.dumps(list(ids), ensure_ascii=False, separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()


def _load_selected_text_rows(path, selected_ids):
    """Stream corpus metadata and retain token arrays only for selected rows.

    The corpus file itself is hash-checked separately. This pass checks basic
    row shape and train/test group separation without materializing/tokenizing
    the full corpus; only selected held rows are retained for evaluation.
    """
    selected_ids = set(selected_ids)
    selected = {}
    groups_by_split = {'train': set(), 'test': set()}
    seen_ids = set()
    with Path(path).open(encoding='utf-8') as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'text JSONL line {line_number} is malformed') from exc
            if not isinstance(row, dict):
                raise ValueError(f'text JSONL line {line_number} must be an object')
            row_id, split, text, groups = (row.get('id'), row.get('split'), row.get('text'),
                                           row.get('source_groups'))
            if not isinstance(row_id, str) or not row_id or row_id in seen_ids:
                raise ValueError(f'text JSONL line {line_number} has a duplicate or empty ID')
            seen_ids.add(row_id)
            if split not in ('train', 'test') or not isinstance(text, str) or not text.strip():
                raise ValueError(f'text JSONL line {line_number} has invalid split/text')
            if not isinstance(groups, list) or not groups or any(not isinstance(g, str) or not g for g in groups):
                raise ValueError(f'text JSONL line {line_number} has invalid source groups')
            groups_by_split[split].update(groups)
            if row_id in selected_ids:
                selected[row_id] = row
    if groups_by_split['train'] & groups_by_split['test']:
        raise ValueError('text corpus source groups cross train/test')
    missing = selected_ids - set(selected)
    if missing:
        raise ValueError(f'selected text rows are missing: {sorted(missing)}')
    return selected


def _cell(value):
    if not isinstance(value, dict):
        raise ValueError('window cell must be an object')
    kind, count = value.get('kind'), value.get('target_tokens')
    if kind not in WINDOW_KINDS or type(count) is not int or count < 2:
        raise ValueError('window cell needs a supported kind and at least two target_tokens')
    return kind, count


def validate_window_selection(manifest, rows, role_codes_by_row, *, source_recipe_sha256,
                              text_data_sha256, tokenizer_sha256, context_limit, open_id,
                              offsets_by_row):
    """Validate exact source-coordinate windows without loading or running a model."""
    if not isinstance(manifest, dict) or manifest.get('schema') != SELECTION_SCHEMA:
        raise ValueError(f'window selection must use {SELECTION_SCHEMA}')
    if manifest.get('source_recipe_sha256') != source_recipe_sha256:
        raise ValueError('window selection belongs to a different source recipe')
    if manifest.get('text_data_sha256') != text_data_sha256:
        raise ValueError('window selection belongs to different text data')
    if manifest.get('tokenizer_sha256') != tokenizer_sha256:
        raise ValueError('window selection belongs to a different tokenizer')
    if manifest.get('split') != 'test':
        raise ValueError('matched AR diagnostic windows must come from the test split')
    if manifest.get('max_context_tokens') != context_limit:
        raise ValueError('window context limit differs from the source recipe')

    by_id = {}
    for row in rows:
        row_id = row.get('id')
        if not isinstance(row_id, str) or not row_id or row_id in by_id:
            raise ValueError('text rows require unique nonempty IDs')
        by_id[row_id] = row
    expected_values = manifest.get('expected_cells')
    selected_values = manifest.get('windows')
    unavailable_values = manifest.get('unavailable_cells')
    if not all(isinstance(value, list) for value in (expected_values, selected_values, unavailable_values)):
        raise ValueError('selection needs expected_cells, windows, and unavailable_cells arrays')
    if not selected_values:
        raise ValueError('at least one matched window is required for an evaluator run')
    expected = [_cell(value) for value in expected_values]
    if any(not isinstance(value, dict) for value in selected_values + unavailable_values):
        raise ValueError('selected and unavailable entries must be objects')
    selected_cells = [_cell(value.get('stratum')) for value in selected_values]
    unavailable_cells = [_cell(value.get('stratum')) for value in unavailable_values]
    if not expected or len(set(expected)) != len(expected):
        raise ValueError('expected_cells must be a nonempty unique list')
    if (len(set(selected_cells)) != len(selected_cells) or len(set(unavailable_cells)) != len(unavailable_cells) or
            set(selected_cells) & set(unavailable_cells) or
            (set(selected_cells) | set(unavailable_cells)) != set(expected)):
        raise ValueError('each expected cell must be selected exactly once or explicitly unavailable')
    for item in unavailable_values:
        if not isinstance(item.get('reason'), str) or not item['reason'].strip():
            raise ValueError('unavailable cells require an explicit reason')

    used_documents, used_groups, window_ids, context_ids = set(), set(), set(), set()
    validated = []
    for item in selected_values:
        if not isinstance(item, dict):
            raise ValueError('window entry must be an object')
        window_id = item.get('window_id')
        if not isinstance(window_id, str) or not window_id or window_id in window_ids:
            raise ValueError('window IDs must be unique nonempty strings')
        window_ids.add(window_id)
        context_id = item.get('context_id')
        if not isinstance(context_id, str) or not context_id or context_id in context_ids:
            raise ValueError('context IDs must be unique nonempty strings')
        context_ids.add(context_id)
        cell = _cell(item.get('stratum'))
        row = by_id.get(item.get('row_id'))
        if row is None:
            raise ValueError(f'{window_id}: unknown source row')
        if row.get('split') != 'test' or item.get('split') != 'test':
            raise ValueError(f'{window_id}: selected source is not held test data')
        text = row.get('text')
        document_sha = hashlib.sha256(text.encode()).hexdigest() if isinstance(text, str) else None
        if document_sha != item.get('document_sha256'):
            raise ValueError(f'{window_id}: source document hash mismatch')
        groups = row.get('source_groups')
        if not isinstance(groups, list) or not groups or item.get('source_groups') != groups:
            raise ValueError(f'{window_id}: source-group binding mismatch')
        if document_sha in used_documents or used_groups.intersection(groups):
            raise ValueError('selected windows must have distinct documents and disjoint source groups')
        used_documents.add(document_sha)
        used_groups.update(groups)

        ids = row.get('token_ids')
        if not isinstance(ids, list) or not ids or any(type(token) is not int or token < 0 for token in ids):
            raise ValueError(f'{window_id}: source row has no valid token IDs')
        offsets = offsets_by_row.get(item['row_id'])
        if not isinstance(offsets, list) or len(offsets) != len(ids):
            raise ValueError(f'{window_id}: tokenizer offsets are missing or misaligned')
        start, end, prefix_start = item.get('target_start'), item.get('target_end'), item.get('prefix_start')
        if type(start) is not int or type(end) is not int or type(prefix_start) is not int:
            raise ValueError(f'{window_id}: token coordinates must be integers')
        if start < 0 or end > len(ids) or end <= start or end - start != cell[1]:
            raise ValueError(f'{window_id}: target span does not match its stratum')
        max_source_prefix = context_limit - cell[1] - 1  # reserve one position for the open control token
        if max_source_prefix < 0:
            raise ValueError(f'{window_id}: target is longer than the context limit')
        expected_prefix_start = max(0, start - max_source_prefix)
        if prefix_start != expected_prefix_start:
            raise ValueError(f'{window_id}: prefix does not preserve the maximum available in-limit context')
        role_codes = role_codes_by_row.get(item['row_id'])
        if not isinstance(role_codes, list) or len(role_codes) != len(ids):
            raise ValueError(f'{window_id}: assistant-role labels are missing or misaligned')
        allowed_role_names = {'assistant_reasoning', 'assistant_reply'} if cell[0] == 'prose' else {'assistant_reply'}
        target_role = item.get('target_role')
        if (target_role not in allowed_role_names or
                any(code != ROLE_CODES.index(target_role) for code in role_codes[start:end])):
            raise ValueError(f'{window_id}: target span is outside its declared assistant role class')
        evidence = item.get('class_evidence')
        allowed_boundaries = {
            'prose': {'assistant_prose_or_reasoning'},
            'code': {'assistant_fenced_code_body', 'serialized_eval_code_argument'},
            'tool_argument': {'assistant_tool_call_payload_outside_code_string'},
        }[cell[0]]
        if not isinstance(evidence, dict) or evidence.get('boundary_type') not in allowed_boundaries:
            raise ValueError(f'{window_id}: target class lacks its expected source-boundary evidence')
        region_start, region_end = evidence.get('region_char_start'), evidence.get('region_char_end')
        target_char_start, target_char_end = evidence.get('target_char_start'), evidence.get('target_char_end')
        if any(type(value) is not int for value in (region_start, region_end, target_char_start, target_char_end)):
            raise ValueError(f'{window_id}: class evidence character coordinates must be integers')
        if not (0 <= region_start <= target_char_start < target_char_end <= region_end <= len(text)):
            raise ValueError(f'{window_id}: class evidence coordinates are outside the source document')
        if hashlib.sha256(text[region_start:region_end].encode()).hexdigest() != evidence.get('region_sha256'):
            raise ValueError(f'{window_id}: class evidence region hash mismatch')
        if hashlib.sha256(text[target_char_start:target_char_end].encode()).hexdigest() != evidence.get('target_text_sha256'):
            raise ValueError(f'{window_id}: class evidence target-text hash mismatch')
        if (region_start, region_end) not in _source_class_regions(text)[cell[0]]:
            raise ValueError(f'{window_id}: declared target class does not match serialized source boundaries')
        if offsets[start][0] != target_char_start or offsets[end - 1][1] != target_char_end:
            raise ValueError(f'{window_id}: tokenizer offsets do not match the declared target characters')
        if any(a < region_start or b > region_end or b <= a for a, b in offsets[start:end]):
            raise ValueError(f'{window_id}: target tokens extend beyond their declared class region')
        prefix_ids = [open_id, *ids[prefix_start:start]]
        target_ids = ids[start:end]
        if item.get('prefix_token_ids_sha256') != token_ids_sha256(prefix_ids):
            raise ValueError(f'{window_id}: prefix token ID hash mismatch')
        if item.get('target_token_ids_sha256') != token_ids_sha256(target_ids):
            raise ValueError(f'{window_id}: target token ID hash mismatch')
        if len(prefix_ids) + len(target_ids) > context_limit:
            raise ValueError(f'{window_id}: assembled window exceeds the context limit')
        validated.append({
            'window_id': window_id, 'context_id': context_id,
            'stratum': {'kind': cell[0], 'target_tokens': cell[1]},
            'row_id': item['row_id'], 'document_sha256': document_sha,
            'source_groups': groups, 'prefix_start': prefix_start, 'target_start': start,
            'target_end': end, 'target_role': target_role,
            'prefix_token_ids': prefix_ids, 'target_token_ids': target_ids,
            'prefix_token_ids_sha256': token_ids_sha256(prefix_ids),
            'target_token_ids_sha256': token_ids_sha256(target_ids),
        })
    return validated


def _special_id(tokenizer, token):
    try:
        value = tokenizer.convert_tokens_to_ids(token)
        return value if isinstance(value, int) and value != tokenizer.unk_token_id else None
    except Exception:
        return None


def _row_role_codes(rows, tokenizer, open_id, close_id):
    start_id = _special_id(tokenizer, '<|im_start|>')
    if start_id is None:
        raise ValueError('tokenizer lacks the chat role marker required by the selection manifest')
    role_ids = {}
    for name in ('system', 'user', 'assistant', 'tool'):
        value = _special_id(tokenizer, name)
        if value is not None:
            role_ids[value] = name
    if _special_id(tokenizer, 'assistant') not in role_ids:
        raise ValueError('tokenizer lacks the assistant role marker required by the selection manifest')
    think_open, think_close = _special_id(tokenizer, '<think>'), _special_id(tokenizer, '</think>')
    return {row['id']: chat_roles([open_id, *row['token_ids'], close_id], start_id=start_id,
                                  role_ids=role_ids, think_open=think_open, think_close=think_close)[1:-1]
            for row in rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-recipe', type=Path, required=True,
                        help='immutable evaluator source pin: parent recipe/checkpoint/heads/text/base model')
    parser.add_argument('--window-selection', type=Path, required=True,
                        help='hashed test-only window manifest with selected and unavailable cells')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise ValueError('fresh immutable diagnostic output required')
    if args.threads < 1:
        raise ValueError('threads must be positive')

    recipe = json.loads(args.source_recipe.read_text())
    source = _verify_source_recipe(recipe, args.source_recipe)
    selection = json.loads(args.window_selection.read_text())
    selection_sha256 = sha(args.window_selection)
    expected_text_sha = recipe['text_data']['sha256']
    if selection.get('source_recipe_sha256') != source['recipe_sha256']:
        raise ValueError('window manifest is not pinned to this exact source recipe')
    if selection.get('text_data_sha256') != expected_text_sha:
        raise ValueError('window manifest is not pinned to the source recipe text data')

    torch.set_num_threads(args.threads)
    # The exact exported serving heads contain the selected backbone deltas and
    # port parameters.  The full training checkpoint above is hash-checked only;
    # it is intentionally never deserialized, so optimizer/RNG state is not loaded.
    from ..serve import load_engine
    engine = load_engine(str(source['base_path']), heads_checkpoint=str(source['pins']['heads_checkpoint']),
                         device=args.device, dtype=torch.bfloat16 if torch.device(args.device).type != 'cpu' else torch.float32)
    engine.backbone.eval()
    engine.heads.eval()
    selected_ids = [item.get('row_id') for item in selection.get('windows', [])
                    if isinstance(item, dict) and isinstance(item.get('row_id'), str)]
    selected_rows = _load_selected_text_rows(source['pins']['text_data'], selected_ids)
    from ..data.text_corpus import tokenizer_fingerprint
    tokenizer_sha256 = tokenizer_fingerprint(engine.tokenizer)
    for row in selected_rows.values():
        ids = row.get('token_ids')
        if (row.get('tokenizer_sha256') != tokenizer_sha256 or not isinstance(ids, list) or not ids or
                any(type(token) is not int or token < 0 or token >= len(engine.tokenizer) for token in ids)):
            raise ValueError(f"selected text row {row.get('id')} has invalid token IDs/tokenizer fingerprint")
    rows = list(selected_rows.values())
    offsets_by_row = {}
    for row in rows:
        encoded = engine.tokenizer(row['text'], add_special_tokens=False, return_offsets_mapping=True)
        if encoded['input_ids'] != row['token_ids']:
            raise ValueError(f"selected text row {row['id']} no longer matches its exact tokenizer IDs")
        offsets_by_row[row['id']] = [tuple(offset) for offset in encoded['offset_mapping']]
    role_codes = _row_role_codes(rows, engine.tokenizer, engine.backbone.controls.open_id,
                                engine.backbone.controls.close_id)
    windows = validate_window_selection(selection, rows, role_codes,
        source_recipe_sha256=source['recipe_sha256'], text_data_sha256=expected_text_sha,
        tokenizer_sha256=tokenizer_sha256, context_limit=source['context_limit'],
        open_id=engine.backbone.controls.open_id, offsets_by_row=offsets_by_row)

    output_rows = []
    for window in windows:
        prefix = torch.tensor([window['prefix_token_ids']], device=args.device, dtype=torch.long)
        span = torch.tensor([window['target_token_ids']], device=args.device, dtype=torch.long)
        scores = autoregressive_history_metrics(engine.backbone, engine.heads, prefix, span,
            steps=span.shape[1], kinds=('ar_greedy', 'ar_projection'))
        output_rows.append({
            'window_id': window['window_id'], 'context_id': window['context_id'],
            'stratum': window['stratum'],
            'target_role': window['target_role'],
            'row_id': window['row_id'], 'document_sha256': window['document_sha256'],
            'source_groups': window['source_groups'], 'split': 'test',
            'prefix_token_count': int(prefix.shape[1]), 'target_token_count': int(span.shape[1]),
            'prefix_token_ids_sha256': window['prefix_token_ids_sha256'],
            'target_token_ids_sha256': window['target_token_ids_sha256'],
            'scores': scores,
        })

    report = {
        'schema': 'natlang.matched-autoregressive-history/1',
        'source_recipe': {'path': str(args.source_recipe), 'sha256': source['recipe_sha256']},
        'training_recipe': {'path': str(source['pins']['training_recipe']), 'sha256': recipe['training_recipe']['sha256']},
        'source_checkpoint': {'path': str(source['pins']['source_checkpoint']),
                              'sha256': recipe['source_checkpoint']['sha256'], 'loaded': False},
        'source_pairing': {'step': recipe['source_pairing']['step'],
                           'freeze_receipt': recipe['source_pairing']['freeze_receipt'],
                           'frozen_best_manifest': recipe['source_pairing']['frozen_best_manifest']},
        'heads_checkpoint': {'path': str(source['pins']['heads_checkpoint']),
                             'sha256': recipe['heads_checkpoint']['sha256'], 'loaded_for_inference': True},
        'text_data': {'path': str(source['pins']['text_data']), 'sha256': expected_text_sha},
        'base_model': {'path': str(source['base_path']), 'revision': recipe['base_model'].get('revision'),
                       'file_count': len(recipe['base_model']['files'])},
        'window_selection': {'path': str(args.window_selection), 'sha256': selection_sha256,
                             'split': selection['split'], 'max_context_tokens': source['context_limit'],
                             'expected_cells': selection['expected_cells'],
                             'selected_count': len(windows), 'unavailable_cells': selection['unavailable_cells']},
        'text_row_validation': {'policy': 'streamed corpus metadata; token IDs retained only for selected test rows',
                                'selected_rows': len(rows)},
        'evaluator_code_snapshot': {'git_commit': source['code_snapshot']['git_commit'],
                                    'file_count': len(source['code_snapshot']['files'])},
        'tokenizer_sha256': tokenizer_sha256,
        'weights_restoration': 'shared serve.load_engine from exact heads export and pinned base model; no optimizer or RNG load',
        'rollout_metric': 'shared eval.projected_history.autoregressive_history_metrics',
        'consumers': ['ar_greedy', 'ar_projection'],
        'windows': output_rows,
        'foundation_qualified': False, 'runtime_qualified': False, 'task_qualified': False,
        'scope': ('Matched held-test windows comparing crisp greedy feedback with full projected feedback. '
                  'Targets after a generated divergence are reference labels, not asserted valid continuations; '
                  'use per-window first-divergence and exact-prefix survival fields. This diagnostic grants no qualification.'),
    }
    args.out.mkdir(parents=True)
    (args.out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'schema': report['schema'], 'selected_windows': len(windows),
                      'unavailable_cells': len(selection['unavailable_cells']),
                      'source_recipe_sha256': source['recipe_sha256'],
                      'window_selection_sha256': selection_sha256,
                      'scope': report['scope']}), flush=True)


if __name__ == '__main__':
    main()
