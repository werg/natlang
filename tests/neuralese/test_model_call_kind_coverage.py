"""Standing guard: every model-facing call kind declares its SFT target, RL action treatment and rendering parity.

The registry is training/model-call-kinds.json. A model call site in ts-host/src carries a `model-call-kind: <id>`
comment on its line or one of the four lines above it; a serving route in serve/http.py is listed under `endpoints`.
An undeclared site or route, a declared kind nothing issues, or a column without evidence fails here, so a new call
kind cannot reach the runtime without its training coverage being declared (decide shipped without it).
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = json.loads((ROOT / 'training/model-call-kinds.json').read_text())
KINDS = REGISTRY['kinds']
SRC = ROOT / 'ts-host/src'
HTTP = ROOT / 'training/neuralese/natlang_neuralese/serve/http.py'
AUDIT = ROOT / REGISTRY['audit']

# Lines that issue a request to the executing model, or author a natlang call the runtime runs on it.
MODEL_ENDPOINT = r'/v1/(?:chat/completions|natlang/score|neuralese/(?:decide|decide_many|view|native-states/))'
SITE_PATTERNS = (
    re.compile(r'\bthis\.driver\('),                     # the agent's own turn
    re.compile(r'\.driver \?\? this\.driver\)\('),       # a turn on an optional second driver
    re.compile(r'\b(?:decide|scorer)\(\{\s*messages'),   # a scoring request
    re.compile(r'fetchModel\(.*' + MODEL_ENDPOINT),       # a direct request to a model endpoint
    re.compile(r'\binvokeDefinition\(.*\{\s*id:|\binvokeDefinition\([^,]*,\s*\{\s*$'),  # an inline runtime-authored call
    re.compile(r"\bbuiltinDefinition\('"),               # a built-in natlang program run by the runtime
)
# Transports carry requests built elsewhere; they are not call kinds.
TRANSPORTS = {'model/chat-completion.ts', 'model/scoring.ts', 'model/neuralese-server.ts', 'model/openai-compatible.ts',
              'model/local-server.ts', 'model/neuralese-info.ts', 'browser'}
ANNOTATION = re.compile(r'model-call-kind:\s*([a-z0-9-]+)')
COLUMNS = ('sft', 'rl', 'parity')


def sites():
    for path in sorted(SRC.rglob('*.ts')):
        rel = path.relative_to(SRC).as_posix()
        if rel.endswith('.d.ts') or rel in TRANSPORTS or rel.split('/')[0] in TRANSPORTS:
            continue
        lines = path.read_text(encoding='utf-8').split('\n')
        for index, line in enumerate(lines):
            if line.lstrip().startswith(('//', '*', '/*')) or not any(p.search(line) for p in SITE_PATTERNS):
                continue
            window = '\n'.join(lines[max(0, index - 4):index + 1])
            yield rel, index + 1, line.strip(), ANNOTATION.findall(window)


def test_the_scan_recognizes_each_form_of_model_call():
    calls = ["response = await this.driver(turnRequest, signal);",
             "const answer = await (review.driver ?? this.driver)({ messages: fork,",
             "scores = await decide({ messages: encoded.messages, options: replies },",
             "const scores = await scorer({ messages, options: ['true', 'false'] }, signal);",
             "const response = await fetchModel(`${base}/v1/neuralese/view`, { method: 'POST',",
             "await fetchModel(url + '/v1/chat/completions', init)",
             "const value = await invokeDefinition(this.runtime.frame!, {",
             "await invokeDefinition({ ...frame }, { id: 'refinement:holds', name: 'holds',",
             "const definition = builtinDefinition('progressJudge');"]
    others = ["return invokeDefinition(frame, definition, args, options);",
              "await fetchModel(base + '/v1/neuralese/info', { headers })",
              "const decide = decisionScorer(this.driver);"]
    assert all(any(p.search(line) for p in SITE_PATTERNS) for line in calls)
    assert not any(any(p.search(line) for p in SITE_PATTERNS) for line in others)


def test_every_model_call_site_declares_a_registered_kind():
    undeclared, unknown, used = [], [], set()
    for rel, number, line, kinds in sites():
        if not kinds:
            undeclared.append(f'{rel}:{number}: {line[:120]}')
        for kind in kinds:
            used.add(kind)
            if kind not in KINDS:
                unknown.append(f'{rel}:{number}: {kind}')
    assert not undeclared, ('model call sites without a `model-call-kind:` annotation; declare the kind in '
                            'training/model-call-kinds.json with its SFT/RL/parity coverage:\n' + '\n'.join(undeclared))
    assert not unknown, 'annotations naming undeclared kinds:\n' + '\n'.join(unknown)
    issued = {kind for kind, entry in KINDS.items() if entry.get('issued_by') != 'server'}
    assert issued <= used, f'declared kinds no call site issues (remove them or annotate the site): {sorted(issued - used)}'


def test_every_serving_route_is_declared():
    routes = set(re.findall(r'"(/v1/[^"]+|/health|/metrics)"', HTTP.read_text(encoding='utf-8')))
    declared = REGISTRY['endpoints']
    assert routes <= set(declared), f'serve/http.py routes missing from training/model-call-kinds.json: {sorted(routes - set(declared))}'
    assert set(declared) <= routes, f'declared routes the server no longer has: {sorted(set(declared) - routes)}'
    for route, entry in declared.items():
        if entry.get('model_facing') is False:
            assert entry.get('reason'), route
            continue
        assert entry.get('kinds') and set(entry['kinds']) <= set(KINDS), route
    for kind, entry in KINDS.items():
        for route in entry['endpoints']:
            assert kind in declared.get(route, {}).get('kinds', []), (kind, route)


def test_every_kind_declares_sft_rl_and_parity():
    audit = AUDIT.read_text(encoding='utf-8')
    anchors = set(re.findall(r'<a id="([a-z0-9-]+)"></a>', audit))
    for kind, entry in KINDS.items():
        assert entry.get('summary') and entry.get('model_output') in ('generated', 'scored-options', 'forced'), kind
        for column in COLUMNS:
            cell = entry[column]
            status = cell.get('status')
            if status == 'covered':
                assert cell.get('evidence'), (kind, column)
                missing = [p for p in cell['evidence'] if not (ROOT / p).exists()]
                assert not missing, f'{kind}.{column} evidence missing: {missing}'
            elif status == 'gap':
                assert cell.get('owner') and cell.get('section') in anchors, \
                    f'{kind}.{column}: a gap names its owner and an anchored section of {REGISTRY["audit"]}'
            elif status == 'not-applicable':
                assert cell.get('reason'), (kind, column)
                assert not (column == 'sft' and entry['model_output'] != 'forced'), \
                    f'{kind}: a model-produced output always has an SFT target (owner: supervise the model\'s own outputs)'
            else:
                raise AssertionError(f'{kind}.{column}: unknown status {status!r}')


def test_guidance_runtime_calls_declare_individual_training_coverage():
    calls = REGISTRY['runtime_calls']
    assert {'selectGuidanceFacts', 'writeGuidance', 'planTeacherAttempts'} <= set(calls)
    anchors = set(re.findall(r'<a id="([a-z0-9-]+)"></a>', AUDIT.read_text(encoding='utf-8')))
    for name, entry in calls.items():
        assert entry['kind'] in KINDS and (ROOT / entry['source']).is_file(), name
        assert name in (ROOT / entry['source']).name
        for column in COLUMNS:
            cell = entry[column]
            if cell['status'] == 'covered':
                assert cell.get('evidence') and all((ROOT / path).exists() for path in cell['evidence']), (name, column)
            else:
                assert cell['status'] == 'gap' and cell.get('owner') and cell.get('section') in anchors, (name, column)
