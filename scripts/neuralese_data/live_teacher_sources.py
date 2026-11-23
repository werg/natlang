"""Admission of observed live Natlang work; effects are verified, never called portable replays."""
import hashlib
import gzip
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from natlang_neuralese.common.hashing import sha256_file_hex
from observed_graph_metrics import graph_metrics

NATIVE_TRAJECTORY_VERSION = 'natlang.teacher_trajectory.native/1'
OBSERVED_WORKSPACE = 'observed-live-workspace'


def open_goal_fingerprint(goal):
    """Hash the JSON input value recorded by the Natlang runtime."""
    encoded = json.dumps(goal, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return len(encoded), hashlib.sha256(encoded).hexdigest()


def same_json_value(left, right):
    """Compare JSON values without Python's bool/number coercion."""
    if type(left) is bool or type(right) is bool:
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return (not isinstance(left, float) or math.isfinite(left)) and (
            not isinstance(right, float) or math.isfinite(right)) and left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same_json_value(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(same_json_value(a, b) for a, b in zip(left, right))
    return left == right


def validate_reassessment_check_roles(proof, original_checks, root):
    """A projection may omit only exhaustively declared, source-bound optional roles."""
    checks = proof.get('checks')
    if (not isinstance(checks, dict) or not checks or any(value is not True for value in checks.values()) or
            not isinstance(original_checks, dict) or not original_checks):
        raise ValueError('completed reassessment needs positive independent required checks')
    if set(checks) == set(original_checks):
        return  # Historical corrected whole-oracle verdict, with every original check retained.
    contract, dimensions = proof.get('outcomeContract'), proof.get('outcomeDimensions')
    if not isinstance(contract, dict) or not isinstance(dimensions, dict):
        raise ValueError('omitted reassessment checks need source-bound outcome roles')
    bindings = [item for item in proof.get('checker_sources') or [] if item.get('role') == 'outcome-contract']
    if len(bindings) != 1 or not same_json_value(json.loads(bound_file(bindings[0], root).read_text()), contract):
        raise ValueError('reassessment outcome roles differ from the bound checker contract')
    names, required = [], []
    for role in ('effects', 'returnedValue', 'incidentalValues'):
        declaration, observed = contract.get(role), dimensions.get(role)
        if role == 'incidentalValues' and declaration is None and observed is None:
            continue
        if not isinstance(declaration, dict) or not isinstance(observed, dict):
            raise ValueError('reassessment check roles must account for every original observation')
        role_names = declaration.get('checks')
        needed = role == 'effects' or (role == 'returnedValue' and declaration.get('required') is True)
        if (not isinstance(role_names, list) or any(not isinstance(name, str) or not name for name in role_names) or
                not isinstance(declaration.get('rationale'), str) or not declaration['rationale'].strip() or
                (role == 'effects' and not role_names) or
                (role == 'returnedValue' and type(declaration.get('required')) is not bool) or
                (role == 'returnedValue' and observed.get('required') is not declaration['required'])):
            raise ValueError('reassessment check roles need explicit requirements and rationales')
        expected = {name: checks.get(name) if needed else original_checks.get(name) for name in role_names}
        if (any(type(value) is not bool for value in expected.values()) or
                not same_json_value(observed.get('checks'), expected) or
                observed.get('passed') is not all(expected.values())):
            raise ValueError('reassessment must preserve optional observations and prove required roles')
        names.extend(role_names)
        if needed:
            required.extend(role_names)
    if len(set(names)) != len(names) or set(names) != set(original_checks) or set(required) != set(checks):
        raise ValueError('reassessment roles must partition the exact original check set')


def validate_return_contribution(item, proof, records):
    """Recheck an authored producer projection and its complete native candidate inventory."""
    c = item.get('contribution')
    if c is None:
        return None
    roles = proof.get('outcomeDimensions') or {}
    failed = {**((roles.get('returnedValue') or {}).get('checks') or {}),
              **((roles.get('incidentalValues') or {}).get('checks') or {})}
    valid_path = lambda path: isinstance(path, list) and all(
        isinstance(key, str) or type(key) is int and 0 <= key <= 9007199254740991 for key in path)
    if (not isinstance(c, dict) or c.get('schema') != 'natlang.application-return-contribution/1' or
            c.get('kind') not in ('root', 'forwarded', 'projection', 'uncertain-projection') or
            not all(isinstance(c.get(key), str) for key in ('root_call_id', 'producer_call_id', 'root_value_json')) or
            hashlib.sha256(c['root_value_json'].encode()).hexdigest() != c.get('root_sha256') or
            not all(valid_path(c.get(key)) for key in ('root_path', 'producer_path')) or
            not isinstance(c.get('checks'), list) or not c['checks'] or
            any(not isinstance(check, str) or failed.get(check) is not False for check in c['checks']) or
            len(c['checks']) != len(set(c['checks'])) or
            not isinstance(c.get('lineage'), list) or not c['lineage'] or
            any(not isinstance(key, str) for key in c['lineage']) or len(c['lineage']) != len(set(c['lineage']))):
        raise ValueError('invalid authored application return contribution')
    try:
        root_value, producer_value = json.loads(c['root_value_json']), json.loads(item['valueJson'])
    except (TypeError, ValueError) as error:
        raise ValueError('invalid application return contribution JSON') from error
    if not same_json_value(root_value, proof.get('actualResult')):
        raise ValueError('application return contribution differs from actual application result')
    root, producer = records.get(c['root_call_id']), records.get(c['producer_call_id'])
    roots = [key for key, row in records.items() if row.get('parent_call_id') is None]
    def matches(record, text, digest):
        output = (record or {}).get('output') or {}
        return ((record or {}).get('outcome') == 'done' and output.get('complete') is True and
                output.get('hash') == digest and output.get('bytes') == len(text.encode()))
    if (roots != [c['root_call_id']] or not root or not producer or
            not matches(root, c['root_value_json'], c['root_sha256']) or
            not matches(producer, item['valueJson'], item['sha256']) or
            any((producer.get('definition') or {}).get(key) != item['definition'].get(key) for key in ('name', 'source')) or
            c['lineage'][0] != root.get('call_id') or c['lineage'][-1] != producer.get('call_id')):
        raise ValueError('application return contribution differs from actual captured outputs')
    def at(value, path):
        for key in path:
            if isinstance(value, dict) and isinstance(key, str) and key in value:
                value = value[key]
            elif isinstance(value, list) and type(key) is int and 0 <= key < len(value):
                value = value[key]
            else:
                return False, None
        return True, value
    def same_projection(value):
        left, right = at(root_value, c['root_path']), at(value, c['producer_path'])
        return left[0] and right[0] and same_json_value(left[1], right[1])
    if c['kind'] == 'root':
        if producer is not root or len(c['lineage']) != 1 or c['root_path'] or c['producer_path']:
            raise ValueError('root contribution must identify its whole actual root return')
        if not same_projection(producer_value):
            raise ValueError('application return projection differs from actual producer value')
        return c['producer_call_id']
    # An empty JSON reply object is present in JavaScript too; do not coerce it to no reply.
    reply_present = lambda turn: isinstance(turn.get('reply'), (dict, list)) or bool(turn.get('reply'))
    def sampled(record):
        execution, cost = record.get('executor') or {}, record.get('cost') or {}
        return (record.get('outcome') == 'done' and execution.get('kind') in ('agent', 'crisp-agent') and
                bool(execution.get('model_id')) and isinstance(cost.get('model_requests'), (int, float)) and
                cost['model_requests'] > 0 and str((record.get('definition') or {}).get('source') or '').endswith('.nl') and
                any(turn.get('kind') == 'generation' and reply_present(turn) and not turn.get('error')
                    for turn in record.get('model_turns') or []))
    def chain(record):
        path, seen = [], set()
        while record:
            key = record.get('call_id')
            if key in seen:
                return None
            seen.add(key)
            path.insert(0, key)
            if key == root['call_id']:
                return path
            record = records.get(record.get('parent_call_id'))
            if not record or (record.get('executor') or {}).get('kind') != 'crisp' or record.get('outcome') != 'done':
                return None
        return None
    if producer is root or not sampled(producer) or chain(producer) != c['lineage']:
        raise ValueError('application return contribution needs a sampled NL producer with actual crisp ancestry')
    if c['kind'] == 'forwarded':
        if c['root_path'] or c['producer_path'] or not same_projection(producer_value):
            raise ValueError('forwarding contribution cannot project values')
        for parent_id in c['lineage'][:-1]:
            parent = records[parent_id]
            if (sum(row.get('parent_call_id') == parent_id for row in records.values()) != 1 or
                    not matches(parent, item['valueJson'], item['sha256'])):
                raise ValueError('forwarding contribution must preserve the sole child whole return')
        return c['producer_call_id']
    suffix, candidates = c.get('producer_source_suffix'), c.get('candidates')
    if not isinstance(suffix, str) or not suffix or not isinstance(candidates, list):
        raise ValueError('projected return contribution needs its authored source and full candidate inventory')
    eligible = {key for key, record in records.items() if sampled(record) and key != root['call_id'] and
                str((record.get('definition') or {}).get('source') or '').endswith(suffix) and chain(record)}
    supplied = [candidate.get('call_id') for candidate in candidates if isinstance(candidate, dict)]
    if len(supplied) != len(candidates) or len(set(supplied)) != len(supplied) or set(supplied) != eligible:
        raise ValueError('application return candidate inventory differs from actual sampled source calls')
    matching = set()
    for candidate in candidates:
        text, digest = candidate.get('value_json'), candidate.get('sha256')
        if (not isinstance(text, str) or hashlib.sha256(text.encode()).hexdigest() != digest or
                not matches(records[candidate['call_id']], text, digest)):
            raise ValueError('application return candidate differs from actual captured output')
        try:
            value = json.loads(text)
        except (TypeError, ValueError) as error:
            raise ValueError('invalid application return candidate JSON') from error
        if same_projection(value):
            matching.add(candidate['call_id'])
    if c['kind'] == 'projection':
        if matching != {c['producer_call_id']} or c.get('uncertainty') is not None:
            raise ValueError('projected return contribution does not identify a unique actual producer')
    elif c.get('uncertainty') == 'ambiguous-actual-producer':
        if len(matching) < 2 or c['producer_call_id'] not in matching:
            raise ValueError('uncertain return contribution does not cover ambiguous actual producers')
    elif c.get('uncertainty') == 'no-verified-actual-producer':
        if matching or c['producer_call_id'] not in eligible:
            raise ValueError('uncertain return contribution contradicts actual projection candidates')
    else:
        raise ValueError('uncertain return contribution needs its actual ambiguity reason')
    return c['producer_call_id']


def validate_uncertain_return_groups(selectors, records):
    """An uncertainty hold cannot omit a possible sampled final-return generation."""
    groups = {}
    for selector in selectors:
        c = selector.get('contribution') or {}
        if c.get('kind') != 'uncertain-projection':
            continue
        key = json.dumps({name: c[name] for name in ('root_call_id', 'producer_source_suffix', 'root_path',
                         'producer_path', 'checks', 'candidates', 'uncertainty')}, sort_keys=True)
        group = groups.setdefault(key, {'selected': set(), 'expected': set()})
        group['selected'].add(c['producer_call_id'])
        root = json.loads(c['root_value_json'])
        def at(value, path):
            for part in path:
                if isinstance(value, dict) and isinstance(part, str) and part in value:
                    value = value[part]
                elif isinstance(value, list) and type(part) is int and 0 <= part < len(value):
                    value = value[part]
                else:
                    return False, None
            return True, value
        root_part = at(root, c['root_path'])
        for candidate in c['candidates']:
            part = at(json.loads(candidate['value_json']), c['producer_path'])
            if c['uncertainty'] == 'no-verified-actual-producer' or (root_part[0] and part[0] and same_json_value(root_part[1], part[1])):
                group['expected'].add(candidate['call_id'])
    if any(group['selected'] != group['expected'] for group in groups.values()):
        raise ValueError('uncertain return hold must cover every possible actual producer')


def declared_return_exclusions(proofs, records):
    """Bind optional wrong returns to original function identity and value bytes."""
    selectors = []
    def contains(value, wanted):
        return same_json_value(value, wanted) or (isinstance(value, dict) and any(contains(item, wanted) for item in value.values())) or (isinstance(value, list) and any(contains(item, wanted) for item in value))
    for proof in proofs:
        dimensions = proof.get('outcomeDimensions') or {}
        incidental, returned = dimensions.get('incidentalValues') or {}, dimensions.get('returnedValue') or {}
        if incidental.get('passed') is not False and returned.get('passed') is not False:
            continue
        values = []
        for role in (incidental, returned):
            selected = role.get('incorrectReturns', [])
            if not isinstance(selected, list):
                raise ValueError('incorrect incidental returns need exact original value bindings')
            for item in selected:
                if item not in values:
                    values.append(item)
        if incidental.get('passed') is False and 'incorrectReturns' not in incidental and not values:
            raise ValueError('incorrect incidental returns need exact original value bindings')
        for item in values:
            definition, text = item.get('definition'), item.get('valueJson')
            if (not isinstance(definition, dict) or not definition.get('name') or not definition.get('source') or
                    not isinstance(text, str) or hashlib.sha256(text.encode('utf-8')).hexdigest() != item.get('sha256')):
                raise ValueError('incorrect return value identity/hash differs')
            try: value = json.loads(text)
            except (TypeError, ValueError) as error: raise ValueError('invalid incorrect return JSON') from error
            if item.get('contribution') is not None:
                validate_return_contribution(item, proof, records)
            elif not contains(proof.get('actualResult'), value):
                raise ValueError('incorrect return value is outside the actual application result')
            selectors.append(item)
        validate_uncertain_return_groups(values, records)
    excluded, matched = {}, set()
    for call_id, record in records.items():
        for index, selector in enumerate(selectors):
            if selector.get('contribution') and call_id != selector['contribution']['producer_call_id']:
                continue
            definition, output = record.get('definition') or {}, record.get('output') or {}
            if (definition.get('name') == selector['definition']['name'] and definition.get('source') == selector['definition']['source'] and
                    output.get('complete') is True and output.get('hash') == selector['sha256'] and output.get('bytes') == len(selector['valueJson'].encode('utf-8'))):
                generations = [i for i, exchange in enumerate(record.get('model_turns') or []) if exchange.get('kind') == 'generation']
                if generations: excluded[call_id] = generations[-1]
                matched.add(index)
    if any(index not in matched and selector.get('required') is not False for index, selector in enumerate(selectors)):
        raise ValueError('incorrect return binding does not match its actual original invocation')
    return excluded


def declared_decision_exclusions(proofs, details):
    """Conservative quality holds bind exact captured call bytes and generation ordinals."""
    calls = {detail['record']['call_id']: (binding, detail['record']) for binding, detail in details}
    excluded = {}
    for proof in proofs:
        review = proof.get('decisionReview')
        if review is None:
            continue
        if not isinstance(review, dict) or review.get('schema') != 'natlang.live-decision-review/1' or not isinstance(review.get('excluded'), list):
            raise ValueError('invalid source-bound decision review')
        for item in review['excluded']:
            call_id, ordinal = item.get('call_id'), item.get('ordinal')
            if call_id not in calls or type(ordinal) is not int or ordinal < 0 or not isinstance(item.get('reason'), str) or not item['reason'].strip():
                raise ValueError('decision review lacks an exact original invocation and reason')
            binding, record = calls[call_id]
            turns = record.get('model_turns') or []
            if (item.get('call_sha256') != binding['sha256'] or
                    not same_json_value(item.get('definition'), record.get('definition')) or ordinal >= len(turns) or
                    turns[ordinal].get('kind') != 'generation' or not isinstance(turns[ordinal].get('reply'), dict)):
                raise ValueError('decision review differs from captured call bytes or generation')
            if (call_id, ordinal) in excluded:
                raise ValueError('duplicate source-bound decision review')
            excluded[call_id, ordinal] = item
    return excluded


def verify_entry(interaction, root, *, partial=False):
    """Bind application inputs to the actual invocation, including historical routers."""
    entry = interaction.get('entry')
    if entry is None:
        entry = {'program': 'skills/open-user-goal/runUserGoal.nl', 'callable': 'runUserGoal',
                 'arguments': {'goal': interaction['goal']}}
    if (not isinstance(entry, dict) or not isinstance(entry.get('program'), str) or
            not entry['program'].endswith(('.nl', '.ts')) or entry['program'].startswith('/') or
            '..' in Path(entry['program']).parts or not entry.get('callable') or
            interaction.get('program') != entry['program'] or not isinstance(entry.get('arguments'), dict)):
        raise ValueError('invalid application entry attestation')
    if interaction.get('entry') is not None:
        source = root.get('definition', {}).get('source')
        program_root = root.get('program_root')
        if (not isinstance(source, str) or not isinstance(program_root, str) or
                str(Path(program_root) / source) != entry['program']):
            raise ValueError('application entry path differs from the recorded source')
    if entry['program'].endswith('.ts') and (
            (root.get('executor') or {}).get('kind') != 'crisp' or
            root.get('definition', {}).get('id') != 'ts:' + str(root.get('definition', {}).get('source')) + '#default'):
        raise ValueError('application entry does not bind its actual TS default export')
    inputs = root.get('inputs') or {}
    arguments = entry['arguments']
    if (root.get('definition', {}).get('name') != entry['callable'] or
            not partial and root.get('outcome') != 'done' or set(inputs) != set(arguments)):
        raise ValueError('open goal does not match the actual root invocation')
    argument_json = entry.get('argumentJson')
    if interaction.get('entry') is not None and (
            not isinstance(argument_json, dict) or set(argument_json) != set(arguments)):
        raise ValueError('application entry needs exact native argument JSON')
    for name, value in arguments.items():
        if argument_json is None:
            size, digest = open_goal_fingerprint(value)  # Historical router's sole string goal.
        else:
            serialized = argument_json[name]
            if not isinstance(serialized, str):
                raise ValueError('application entry argument JSON must be exact text')
            try:
                def reject_constant(constant):
                    raise ValueError(f'nonfinite JSON argument: {constant}')
                captured = json.loads(serialized, parse_constant=reject_constant)
                encoded = serialized.encode('utf-8')
            except (ValueError, UnicodeError) as error:
                raise ValueError('application entry argument JSON is invalid') from error
            if not same_json_value(captured, value):
                raise ValueError('application entry argument JSON differs from the declared input')
            size, digest = len(encoded), hashlib.sha256(encoded).hexdigest()
        ref = inputs.get(name) or {}
        if ref.get('complete') is not True or ref.get('bytes') != size or ref.get('hash') != digest:
            raise ValueError('open goal application input differs from the actual root invocation')


def verify_application_return(binding, records, roots, entry=None):
    """Validate a selected domain value against the actual authored return bytes."""
    if not isinstance(binding, dict) or binding.get('root_call_id') not in roots or len(roots) != 1:
        raise ValueError('application return does not identify its actual authored root')
    text = binding.get('valueJson')
    if not isinstance(text, str):
        raise ValueError('application return needs exact native JSON')
    try:
        value = json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except ValueError as error:
        raise ValueError('invalid application return JSON') from error
    output = records[binding['root_call_id']].get('output') or {}
    encoded = text.encode('utf-8')
    if output.get('complete') is not True or output.get('bytes') != len(encoded) or output.get('hash') != hashlib.sha256(encoded).hexdigest():
        raise ValueError('application return differs from original root output')
    if binding.get('kind') == 'void':
        record = records[binding['root_call_id']]
        definition = record.get('definition') or {}
        if (not isinstance(entry, dict) or entry.get('resultType') != 'void' or
                not str(entry.get('program', '')).endswith('.ts') or definition.get('returns') != 'void' or
                (record.get('executor') or {}).get('kind') != 'crisp' or record.get('outcome') != 'done' or
                text != 'null' or 'selectedResult' not in binding or binding['selectedResult'] is not None):
            raise ValueError('void application return does not bind an actual completed TS void contract')
        return
    if binding.get('kind') not in (None, 'value'):
        raise ValueError('unknown application return kind')
    selected = binding.get('selectedResult')
    results = value.get('results') if isinstance(value, dict) else None
    if ('selectedResult' not in binding or not (same_json_value(value, selected) or
            isinstance(results, list) and len(results) == 1 and isinstance(results[0], dict) and
            'report' in results[0] and same_json_value(results[0]['report'], selected))):
        raise ValueError('selected domain return differs from actual application result')


def verified_training_scope(verification, records, root):
    """Admit independently attested invocations while retaining the complete original graph."""
    scope = verification.get('training_scope')
    if scope is None:
        if verification['status'] != 'passed':
            raise ValueError('partial live verification needs an explicit training scope')
        return None
    if (verification['status'] != 'partial' or not isinstance(scope, dict) or
            scope.get('schema') != 'natlang.training-scope/1'):
        raise ValueError('invalid partial live training scope')
    if scope.get('assessor'):
        bound_file(scope['assessor'], root)
    original = json.loads(bound_file(scope['original_outcome'], root).read_text())
    if original.get('passed') is not False:
        raise ValueError('partial recovery must preserve the original failed outcome')
    selected = {}
    for invocation in scope.get('invocations') or []:
        call_id = invocation.get('call_id')
        checks = invocation.get('checks')
        evidence = invocation.get('evidence')
        if (call_id not in records or call_id in selected or
                records[call_id].get('outcome') != 'done' and invocation.get('ordinals') is None or
                not isinstance(checks, dict) or not checks or any(value is not True for value in checks.values()) or
                not isinstance(evidence, list) or not evidence):
            raise ValueError('recovered invocation lacks independent successful outcome evidence')
        for binding in evidence:
            if binding.get('kind') not in {'workspace', 'cdp'}:
                raise ValueError('recovered invocation needs workspace or CDP evidence')
            bound_file(binding, root)
        ordinals = invocation.get('ordinals')
        if ordinals is not None:
            exchanges = records[call_id].get('model_turns') or []
            if (not isinstance(ordinals, list) or not ordinals or len(set(ordinals)) != len(ordinals) or
                    any(type(index) is not int or index < 0 or index >= len(exchanges) or
                        exchanges[index].get('kind') != 'generation' for index in ordinals)):
                raise ValueError('recovered invocation has invalid generation ordinals')
        selected[call_id] = None if ordinals is None else set(ordinals)
    if not selected:
        raise ValueError('partial live training scope is empty')
    return selected


def bound_file(binding, root):
    path = Path(binding['path'])
    if not path.is_absolute():
        path = root / path
    if sha256_file_hex(path) != binding.get('sha256'):
        raise ValueError(f'live source artifact changed: {path}')
    return path


def verify_original_rejection(packet, proof, root):
    """Keep native failure and a later rejected review as distinct original facts."""
    passed = packet.get('result', {}).get('passed')
    bindings = [item for item in proof.get('evidence', []) if item.get('role') == 'original-rejected-review']
    if not bindings:
        return passed is False
    if len(bindings) != 1 or not isinstance(passed, bool):
        return False
    review = json.loads(bound_file(bindings[0], root).read_text())
    observed = packet.get('execution', {}).get('diagnostics', {}).get('openGoalVerification', {})
    original_input = (observed.get('preparedContentProject') or {}).get('artifact', {}).get('semanticReviewInput') or observed.get('semanticReviewInput')
    exchanges = [review.get('exchange'), *(review.get('exchanges') or [])]
    captured_exchange = any(isinstance(item, dict) and isinstance(item.get('request'), dict) and
                            isinstance(item.get('reply'), dict) for item in exchanges)
    return (review.get('schema') == 'natlang.live-workflow-review/1' and
            (review.get('verdict') or {}).get('accepted') is False and
            isinstance(original_input, dict) and bool(original_input) and same_json_value(review.get('input'), original_input) and
            captured_exchange)


def verify_completed_reassessment(verification, spec, root):
    """Bind a later whole-outcome verdict to immutable original execution and checker evidence."""
    reassessment = verification.get('reassessment')
    if reassessment is None:
        return
    if verification.get('status') != 'passed' or verification.get('training_scope') is not None:
        raise ValueError('completed reassessment cannot relabel a partial training scope')
    original = json.loads(bound_file(reassessment['original_outcome'], root).read_text())
    assessment = json.loads(bound_file(reassessment['assessment'], root).read_text())
    validator = json.loads(bound_file(verification['validator'], root).read_text())
    packet_path = bound_file(validator['original_packet'], root)
    packet_bytes = packet_path.read_bytes()
    packet = json.loads(gzip.decompress(packet_bytes) if packet_path.suffix == '.gz' else packet_bytes)
    native = packet['execution']['diagnostics']['natlangTeacher']
    original_proof = packet['execution']['diagnostics']['openGoalVerification']
    original_calls = native.get('calls') or []
    captured_calls = [json.loads(bound_file(binding, root).read_text()) for binding in spec['calls']]
    if (len(original_calls) != len(captured_calls) or not same_json_value(
            {item['callId']: item for item in original_calls}, {item['callId']: item for item in captured_calls}) or
            not same_json_value(native.get('rootCallIds'), verification.get('root_call_ids')) or
            not same_json_value(native.get('entry'), (spec.get('interaction') or {}).get('entry')) or
            original_proof.get('goal') != (spec.get('interaction') or {}).get('goal')):
        raise ValueError('completed reassessment does not bind the exact original application call graph')
    proofs = [json.loads(bound_file(item, root).read_text()).get('proof') for item in verification['evidence'] if item.get('kind') == 'workspace']
    if len(proofs) != 1 or not isinstance(proofs[0], dict):
        raise ValueError('completed reassessment needs one bound independent whole-outcome proof')
    proof = proofs[0]
    original_completed = json.loads(bound_file(validator['completed_outcome'], root).read_text())
    def normalize_bindings(value):
        normalized = dict(value)
        for field in ('evidence', 'checker_sources'):
            normalized[field] = [{key: item[key] for key in item if key != 'path'} for item in value.get(field) or []]
        return normalized
    if (assessment.get('completed_outcome', {}).get('sha256') != validator['completed_outcome']['sha256'] or
            not same_json_value(normalize_bindings(proof), normalize_bindings(original_completed))):
        raise ValueError('completed reassessment differs from its bound independent verdict')
    if (not verify_original_rejection(packet, proof, root) or not same_json_value(original, packet['result']) or
            assessment.get('schema') != 'natlang.live-reassessment/1' or
            assessment.get('task_id') != spec['id'] or assessment.get('original_sha256') != hashlib.sha256(packet_bytes).hexdigest() or
            proof.get('schema') != 'natlang.live-outcome-reassessment/1' or proof.get('passed') is not True or
            proof.get('task_id') != spec['id'] or proof.get('original_sha256') != assessment['original_sha256'] or
            proof.get('goal') != original_proof.get('goal') or not same_json_value(proof.get('entry'), native.get('entry')) or
            not same_json_value(proof.get('actualResult'), native.get('report')) or
            not isinstance(proof.get('checks'), dict) or
            not proof['checks'] or any(value is not True for value in proof['checks'].values()) or
            not proof.get('rationale') or not proof.get('evidence') or not proof.get('checker_sources')):
        raise ValueError('completed reassessment differs from the original execution or independent whole outcome')
    validate_reassessment_check_roles(proof, original_proof.get('checks'), root)
    for binding in proof['evidence'] + proof['checker_sources']:
        bound_file(binding, root)


def failed_action_turns(detail):
    """Generation ordinals containing failed actions, joined only by original tool-call IDs."""
    errors = [event for event in detail.get('events') or []
              if event.get('kind') == 'action' and event.get('outcome') in {'error', 'rejected', 'blocked', 'budget'}]
    if not errors:
        return {}
    failed_ids = {event.get('tool_call_id') for event in errors}
    if any(not isinstance(ident, str) or not ident for ident in failed_ids):
        raise ValueError('failed live action lacks its original tool-call ID')
    owners = {}
    for index, exchange in enumerate(detail['record'].get('model_turns') or []):
        if exchange.get('kind') == 'generation':
            for call in (exchange.get('reply') or {}).get('raw_calls') or []:
                owners.setdefault(call.get('id'), []).append(index)
    excluded = {}
    for ident in sorted(failed_ids):
        if len(owners.get(ident, [])) != 1:
            raise ValueError('failed live action does not match exactly one captured generation')
        excluded.setdefault(owners[ident][0], []).append(ident)
    return excluded


def observed_training_turns(trajectory):
    """Project the supervised executor decisions from an observed native trajectory."""
    if (trajectory.get('version') != NATIVE_TRAJECTORY_VERSION or
            trajectory.get('provenance', {}).get('execution_kind') != OBSERVED_WORKSPACE or
            trajectory.get('outcome', {}).get('accepted') is not True):
        raise ValueError('unsupported observed workspace trajectory')
    turns = []
    for decision in trajectory['trajectory']:
        if not decision['training_admission']['approved']:
            continue
        if decision.get('phase') == 'decision_reasoning':
            continue   # a reasoned readout's reasoning generation: its free reply was never used (observed_decision)
        assistant = decision['assistant']
        target = {'role': 'assistant', 'content': assistant['content']}
        if assistant['reasoning']:
            target['reasoning_content'] = assistant['reasoning']
        if assistant['raw_calls']:
            target['tool_calls'] = assistant['raw_calls']
        invocation = decision['invocation_id']
        parent = decision['parent_invocation_id']
        turns.append({'messages': decision['context'], 'tools': decision['tools_offered'], 'target': target,
                      'invocation_id': invocation, 'function': decision['function'],
                      'definition_id': decision['definition_id'], 'parent_invocation': parent,
                      'invocation_role': 'child' if parent else 'root', 'live_exchange': decision['live_exchange']})
    return turns


def observed_decision(binding, record, ancestry, index, exchange, assistant=None, denial=None):
    invocation = record['call_id']
    request = exchange.get('request') or {}
    gateway = ((exchange.get('reply') or {}).get('raw_response') or {}).get('vibestudio')
    return {
        # a reasoned readout's reasoning generation (request metadata ``decision_reasoning``) is no action
        'phase': ('decision_reasoning' if isinstance(request.get('decision_reasoning'), dict) else
                  'action' if exchange['kind'] == 'generation' else exchange['kind']),
        'invocation_id': invocation, 'parent_invocation_id': record['parent_call_id'],
        'function': record['definition']['name'], 'definition_id': record['definition']['id'],
        'context': request.get('messages'), 'tools_offered': request.get('tools') or [],
        'assistant': assistant, 'error': exchange.get('error'),
        'training_admission': {'approved': denial is None, **({'reason': denial} if denial else {})},
        'live_exchange': {'call_artifact': binding, 'ordinal': index, 'gateway': gateway,
                          'root_call_id': ancestry[invocation][0], 'call_ancestry': ancestry[invocation],
                          'target_source': 'actual-model-reply', 'student_sampling_claimed': False},
    }


def observed_graph(details, ancestry, scopes, exclusions):
    """Keep each invocation's own event ordering and original resource receipts."""
    return [
        {'call_id': record['call_id'], 'parent_call_id': record['parent_call_id'],
         'parent_action_index': record.get('parent_action_index'), 'program_root': record.get('program_root'),
         'definition': record['definition'], 'executor': record['executor'], 'outcome': record.get('outcome'),
         'root_call_id': ancestry[record['call_id']][0], 'artifact': binding, 'scope': scopes[record['call_id']],
         'inputs': record.get('inputs'), 'output': record.get('output'), 'effects': record.get('effects') or [],
         'started_at': record.get('started_at'), 'ended_at': record.get('ended_at'),
         'graph_custody': record.get('graph_custody'), 'events': detail.get('events') or [],
         'excluded_failed_action_turns': {str(k): v for k, v in exclusions.get(record['call_id'], {}).items()}}
        for binding, detail in details for record in [detail['record']]
    ]


def verify_observed_child_inventory(records):
    if not any(record.get('graph_custody') is not None for record in records.values()):
        return
    children = defaultdict(set)
    for call_id, record in records.items():
        children[record['parent_call_id']].add(call_id)
    for call_id, record in records.items():
        custody = record.get('graph_custody')
        declared = custody.get('child_call_ids') if isinstance(custody, dict) else None
        if (not isinstance(declared, list) or not all(isinstance(child, str) and child for child in declared) or
                custody.get('schema') != 'natlang.call-graph-chronology/1' or
                not record.get('task_id') or custody.get('task_id') != record['task_id'] or
                not isinstance(custody.get('sha256'), str) or not re.fullmatch(r'[0-9a-f]{64}', custody['sha256']) or
                len(declared) != len(children[call_id]) or set(declared) != children[call_id] or
                any(records[child].get('task_id') != record['task_id'] for child in children[call_id])):
            raise ValueError('observed workspace call graph differs from recorded child inventory')


def load_episode(path):
    path = Path(path).resolve()
    spec = json.loads(path.read_text())
    if spec.get('schema') != 'natlang.live-teacher-task/1':
        raise ValueError('unsupported live teacher task manifest')
    for field in ('id', 'family', 'source_namespace'):
        if not isinstance(spec.get(field), str) or not spec[field]:
            raise ValueError(f'live task needs {field}')
    for field in ('source_ids', 'source_groups', 'source_revisions'):
        if not isinstance(spec.get(field), list) or not spec[field] or any(not isinstance(x, str) or not x for x in spec[field]):
            raise ValueError(f'live task needs {field}')
    if spec.get('split') != 'train':
        raise ValueError('live source is not a training task')
    if spec.get('learner_kinds') != ['natlang-executor']:
        raise ValueError('select actual Natlang executor generations explicitly')
    if (spec.get('code') or {}).get('dirty') is not False or not (spec.get('code') or {}).get('git_commit'):
        raise ValueError('live source needs clean capture-code provenance')
    world = spec.get('world') or {}
    if not world.get('run_id') or not world.get('instance'):
        raise ValueError('live task must identify the actual test world and outer run')
    verification = json.loads(bound_file(spec['verification'], path.parent).read_text())
    if (verification.get('schema') != 'natlang.live-task-verification/1' or
            verification.get('task_id') != spec['id'] or verification.get('run_id') != world['run_id'] or
            verification.get('split', spec['split']) != spec['split'] or
            verification.get('world') != world or
            verification.get('status') not in {'passed', 'partial'}):
        raise ValueError('live task verification does not attest this actual task/run')
    checks = verification.get('checks')
    if not isinstance(checks, dict) or not checks or any(v is not True for v in checks.values()):
        raise ValueError('independent live task checks did not all pass')
    bound_file(verification['validator'], path.parent)
    verify_completed_reassessment(verification, spec, path.parent)
    kinds = set()
    incorrect_optional_return = False
    workspace_proofs = []
    application_returns = []
    for evidence in verification.get('evidence') or []:
        evidence_path = bound_file(evidence, path.parent)
        kinds.add(evidence.get('kind'))
        if evidence.get('kind') == 'workspace':
            payload = json.loads(evidence_path.read_text())
            if 'applicationReturn' in payload:
                application_returns.append(payload['applicationReturn'])
            proof = payload.get('proof') or {}
            workspace_proofs.append(proof)
            returned = (proof.get('outcomeDimensions') or {}).get('returnedValue') or {}
            if returned.get('required') is False and returned.get('passed') is False:
                incorrect_optional_return = True
    required = set(spec.get('required_effect_kinds') or [])
    if not required or not required <= {'workspace', 'cdp'} or not required <= kinds:
        raise ValueError('missing actual workspace/CDP effect readback evidence')
    scopes = {}
    for scope in verification.get('calls') or []:
        if scope.get('call_id') in scopes:
            raise ValueError('duplicate call in live validation scope')
        if any(not scope.get(k) for k in ('call_id', 'owner_kind', 'owner_key', 'run_id')):
            raise ValueError('live validation needs exact authoritative eval ownership')
        scopes[scope['call_id']] = scope
    turns, seen, models = [], set(), set()
    records, details = {}, []
    for binding in spec.get('calls') or []:
        detail = json.loads(bound_file(binding, path.parent).read_text())
        record, summary = detail['record'], detail['summary']
        call_id = record['call_id']
        if call_id in seen or call_id not in scopes or detail.get('callId') != call_id:
            raise ValueError('live call is duplicate or outside the verified workflow')
        seen.add(call_id)
        if (summary.get('call_id') != call_id or not summary.get('definition_id') or
                summary['definition_id'] != (record.get('definition') or {}).get('id')):
            raise ValueError('exported call identity differs from original record')
        if any(summary.get(k) != scopes[call_id].get(k) for k in ('owner_kind', 'owner_key', 'run_id')):
            raise ValueError('live call owner/eval run differs from verified scope')
        if (record.get('version') != 'natlang.calls/1' or
                record.get('executor', {}).get('kind') not in ('agent', 'crisp', 'crisp-agent')):
            raise ValueError('live source must contain original Natlang calls')
        if ('parent_call_id' not in record or record['parent_call_id'] is not None and
                (not isinstance(record['parent_call_id'], str) or not record['parent_call_id'])):
            raise ValueError('live call needs its original explicit parent identity')
        if summary.get('parent_call_id') != record['parent_call_id']:
            raise ValueError('live call parent differs from authoritative summary')
        records[call_id] = record
        details.append((binding, detail))
    if seen != set(scopes):
        raise ValueError('workflow call coverage incomplete')
    training_scope = verified_training_scope(verification, records, path.parent)
    roots = {call_id for call_id, record in records.items() if record['parent_call_id'] is None}
    attested_roots = verification.get('root_call_ids')
    if (not isinstance(attested_roots, list) or not attested_roots or
            len(attested_roots) != len(set(attested_roots)) or roots != set(attested_roots)):
        raise ValueError('live call roots differ from independently verified roots')
    interaction = spec.get('interaction')
    entry = (interaction or {}).get('entry')
    if isinstance(entry, dict) and entry.get('resultType') == 'void' and (
            len(application_returns) != 1 or application_returns[0].get('kind') != 'void'):
        raise ValueError('void application entry needs its explicit actual return witness')
    for binding in application_returns:
        verify_application_return(binding, records, roots, entry)
    if interaction != verification.get('interaction'):
        raise ValueError('live task interaction differs from independent verification')
    if interaction is not None:
        if (not isinstance(interaction, dict) or interaction.get('kind') != 'open-user-goal' or
                not isinstance(interaction.get('goal'), str) or not interaction['goal'].strip() or
                not isinstance(interaction.get('program'), str) or len(roots) != 1):
            raise ValueError('invalid open user goal attestation')
        root = records[attested_roots[0]]
        verify_entry(interaction, root, partial=training_scope is not None)
        verify_observed_child_inventory(records)
    ancestry = {}
    for call_id in records:
        chain, current = [], call_id
        while current is not None:
            if current in chain:
                raise ValueError('live call parent cycle')
            if current not in records:
                raise ValueError('live call parent missing from complete workflow tree')
            chain.append(current)
            current = records[current]['parent_call_id']
        ancestry[call_id] = list(reversed(chain))
    incorrect_returns = declared_return_exclusions(workspace_proofs, records)
    reviewed_decisions = declared_decision_exclusions(workspace_proofs, details)
    if interaction is not None and incorrect_optional_return:
        for call_id in roots:
            generations = [index for index, exchange in enumerate(records[call_id].get('model_turns') or [])
                           if exchange.get('kind') == 'generation']
            if generations:
                incorrect_returns[call_id] = generations[-1]
    exclusions, decisions = {}, []
    for binding, detail in details:
        record = detail['record']
        call_id = record['call_id']
        selected_executor = record['executor']['kind'] != 'crisp' and (record.get('outcome') == 'done' or
            training_scope is not None and call_id in training_scope and training_scope[call_id] is not None)
        if not selected_executor and interaction is None:
            continue  # exact mechanism and failed attempts remain evidence, never successful demonstration targets
        exclusions[call_id] = failed_action_turns(detail)
        actions = [event for event in detail.get('events') or [] if event.get('kind') == 'action']
        counts = Counter(event.get('tool_call_id') for event in actions)
        successful_actions = {event.get('tool_call_id') for event in actions
                              if counts[event.get('tool_call_id')] == 1 and event.get('outcome') in {'ok', 'completed'}}
        for index, exchange in enumerate(record.get('model_turns') or []):
            if exchange.get('schema') != 'natlang.model-exchange/1':
                raise ValueError('unsupported live model exchange')
            if exchange.get('kind') != 'generation' or exchange.get('error'):
                if interaction is not None:
                    decisions.append(observed_decision(binding, record, ancestry, index, exchange,
                                                       denial='score or interrupted model exchange'))
                continue
            request, reply = exchange['request'], exchange.get('reply')
            if not isinstance(reply, dict) or not isinstance(request.get('messages'), list):
                raise ValueError('live generation lacks actual request/reply')
            gateway = (reply.get('raw_response') or {}).get('vibestudio') or {}
            attribution = gateway.get('attribution') or {}
            original = gateway.get('response') or {}
            if (not gateway.get('model') or not gateway.get('api') or not gateway.get('turnId') or
                    not attribution.get('callerId') or not attribution.get('callerKind') or
                    not gateway.get('context') or original.get('role') != 'assistant' or
                    not isinstance(original.get('content'), list)):
                raise ValueError('live generation lacks original gateway identity/context/response')
            blocks = original['content']
            original_calls = [block for block in blocks if block.get('type') == 'toolCall']
            calls = reply.get('raw_calls')
            if not isinstance(calls, list) or len(calls) != len(original_calls):
                raise ValueError('live tool target lacks original assistant tool-call IDs/arguments')
            for raw, actual in zip(calls, original_calls):
                function = raw.get('function') or {}
                if (not actual.get('id') or raw.get('id') != actual['id'] or
                        function.get('name') != actual.get('name') or
                        json.loads(function.get('arguments', 'null')) != actual.get('arguments')):
                    raise ValueError('normalized live calls differ from original assistant response')
            if [[c['name'], c['arguments']] for c in original_calls] != (reply.get('calls') or []):
                raise ValueError('normalized live calls differ from original assistant response')
            text = ''.join(block['text'] for block in blocks if block.get('type') == 'text')
            reasoning = '\n'.join(block['thinking'] for block in blocks if block.get('type') == 'thinking')
            if (reply.get('text') or '') != text or (reply.get('reasoning') or '') != reasoning:
                raise ValueError('normalized live text/reasoning differs from original assistant response')
            target = {'role': 'assistant', 'content': text}
            if reasoning:
                target['reasoning_content'] = reasoning
            if calls:
                target['tool_calls'] = calls
            if interaction is not None:
                denial = ('invocation lacks independent outcome evidence' if training_scope is not None and call_id not in training_scope else
                          'decision lacks independent outcome evidence' if training_scope is not None and training_scope[call_id] is not None and index not in training_scope[call_id] else
                          'source-bound decision quality review' if (call_id, index) in reviewed_decisions else
                          'application return failed independent outcome checks'
                          if incorrect_returns.get(call_id) == index else
                          'executor did not complete' if not selected_executor else
                          'decision contains a failed action' if index in exclusions[call_id] else
                          'decision contains an action without execution evidence'
                          if any(call['id'] not in successful_actions for call in original_calls) else
                          'empty model response' if not text and not calls else None)
                decisions.append(observed_decision(
                    binding, record, ancestry, index, exchange,
                    {'content': text, 'reasoning': reasoning, 'calls': [
                        {'id': call['id'], 'source_tool': call['name'], 'arguments': call['arguments']}
                        for call in original_calls],
                     'raw_calls': calls}, denial))
                models.add(gateway['model'])
                continue
            if training_scope is not None and call_id not in training_scope:
                continue
            if training_scope is not None and training_scope[call_id] is not None and index not in training_scope[call_id]:
                continue
            if index in exclusions[call_id]:
                continue  # failed action remains raw evidence and later zero-weight history, never a target
            if not target['content'] and not target.get('tool_calls'):
                continue
            models.add(gateway['model'])
            definition = record.get('definition') or {}
            turns.append({'messages': request['messages'], 'tools': request.get('tools') or [], 'target': target,
                'invocation_id': call_id, 'function': definition.get('name') or 'unknown',
                'definition_id': definition.get('id'), 'parent_invocation': record.get('parent_call_id'),
                'invocation_role': 'child' if record.get('parent_call_id') else 'root',
                'live_exchange': {'call_artifact': binding, 'ordinal': index, 'gateway': gateway,
                                  'root_call_id': ancestry[call_id][0], 'call_ancestry': ancestry[call_id],
                                  'target_source': 'actual-model-reply', 'student_sampling_claimed': False}})
    task = {key: spec[key] for key in ('id', 'family', 'split', 'source_namespace', 'source_ids', 'source_groups', 'source_revisions')}
    task['source'] = OBSERVED_WORKSPACE
    source = {'path': str(path), 'sha256': sha256_file_hex(path)}
    episode_key = f'live:{spec["id"]}:{source["sha256"]}'
    trajectory = None
    if interaction is not None:
        task['interaction'] = interaction
        trajectory = {
            'version': NATIVE_TRAJECTORY_VERSION, 'id': episode_key, 'task': task,
            'provenance': {'collection_role': 'teacher', 'execution_kind': OBSERVED_WORKSPACE,
                           'code': spec['code'], 'model_id': ','.join(sorted(models)),
                           'source_manifest': source, 'verification': spec['verification']},
            'outcome': {'accepted': True, 'status': verification['status'], 'world': world, 'checks': checks,
                        'task_completed': verification['status'] == 'passed',
                        'root_call_ids': attested_roots,
                        **({'reassessment': verification['reassessment']} if verification.get('reassessment') else {}),
                        **({'excluded_return_decisions': [{'invocation_id': call_id, 'ordinal': ordinal}
                             for call_id, ordinal in incorrect_returns.items()]} if incorrect_returns else {}),
                        **({'excluded_review_decisions': list(reviewed_decisions.values())} if reviewed_decisions else {}),
                        **({'training_scope': verification['training_scope']} if training_scope is not None else {}),
                        'graph_metrics': graph_metrics(records.values()),
                        'invocation_ledger': observed_graph(details, ancestry, scopes, exclusions)},
            'trajectory': decisions,
        }
        turns = observed_training_turns(trajectory)
    if not turns and not (trajectory is not None and (incorrect_optional_return or incorrect_returns or reviewed_decisions)):
        raise ValueError('workflow call coverage incomplete or has no generated executor targets')
    return {'key': episode_key, 'source': 'teacher', 'run': str(path),
            'task_id': spec['id'], 'family': spec['family'], 'task': task, 'turns': turns,
            'teacher': {'model_id': ','.join(sorted(models))},
            'verification_mode': 'observed-live', 'live_source': source,
            **({'trajectory_ir': trajectory} if trajectory is not None else {}),
            'world': world, 'call_tree': [
                {'call_id': r['call_id'], 'parent_call_id': r['parent_call_id'], 'executor': r['executor']['kind'],
                 'outcome': r.get('outcome'), 'root_call_id': ancestry[r['call_id']][0], 'artifact': binding,
                 'scope': scopes[r['call_id']],
                 'excluded_failed_action_turns': {str(k): v for k, v in exclusions.get(r['call_id'], {}).items()}}
                for binding, detail in details for r in [detail['record']]]}
