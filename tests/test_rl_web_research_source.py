"""web-research RL source (scripts/rl_tasks/sources/web_research.py): an item's recorded search results and page readings
become its frozen corpus, the `web` service reproduces every recorded visit and ranks the corpus for any query (CJK
included), and web_research_replay.mjs reaches the gold through the teacher's own calls and stays silent on an empty
corpus. Items whose teacher answer misses the gold, or whose gold is in no recorded output, are not admitted."""
import io
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest
import zstandard

from scripts.rl_tasks.sources import web_research

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / 'ts-host'


def _node():
    found = sorted(Path.home().glob('.local/share/natlang-dev/node-v24*/bin/node'), reverse=True)
    return str(found[0]) if found else shutil.which('node')


def _call(name, **arguments):
    return '<tool_call>\n' + json.dumps({'name': name, 'arguments': arguments}, ensure_ascii=False) + '\n</tool_call>'


def _responses(*bodies):
    return '\n'.join(f'<tool_response>\n{b}\n</tool_response>' for b in bodies)


def _search_output(query, entries):
    lines = [f"A Google search for '{query}' found {len(entries)} results:\n\n## Web Results"]
    for k, (title, url, snippet) in enumerate(entries, 1):
        lines.append(f'{k}. [{title}]({url})\n{snippet}\n')
    return '\n'.join(lines)


def _visit_output(url, goal, text):
    return f'The useful information in {url} for user goal {goal} as follows: \n\nEvidence in page: \n{text}\n\nSummary: \n{text}\n\n'


def _row(question, gold, final, search, visit, sub=None):
    (query, entries), (url, goal, text) = search, visit
    messages = [
        {'role': 'system', 'content': 'You are a research lead.'},
        {'role': 'user', 'content': question},
        {'role': 'assistant', 'content': '<think>plan</think>\n' + _call('search', query=[query])},
        {'role': 'user', 'content': _responses(_search_output(query, entries))},
        {'role': 'assistant', 'content': _call('visit', url=[url], goal=goal)},
        {'role': 'user', 'content': _responses(_visit_output(url, goal, text))},
        {'role': 'assistant', 'content': f'<think>done</think>\n<answer>{final}</answer>'},
    ]
    return {'source': 'redsearcher', 'question': question, 'answer': gold, 'messages': messages,
            'subagents': sub or [], 'num_subagents': len(sub or [])}


def _write(path, rows):
    with open(path, 'wb') as raw:
        writer = zstandard.ZstdCompressor().stream_writer(raw)
        for row in rows:
            writer.write((json.dumps(row, ensure_ascii=False) + '\n').encode())
        writer.close()


ROWS = [
    _row('Which river flows through the city where the painter of The Night Watch was born?', 'Oude Rijn',
         'The river is the Oude Rijn.',
         ('Night Watch painter birthplace', [('Rembrandt - Wikipedia', 'https://en.wikipedia.org/wiki/Rembrandt',
                                              'Rembrandt was born in Leiden, in the Dutch Republic.'),
                                             ('Leiden', 'https://en.wikipedia.org/wiki/Leiden', 'Leiden is a city.')]),
         ('https://en.wikipedia.org/wiki/Leiden', 'which river flows through Leiden',
          'Leiden lies on the Oude Rijn,\u2028a branch of the Rhine.')),
    _row('《夜巡》作者出生的城市位于哪条河畔？', '旧莱茵河', '旧莱茵河',
         ('伦勃朗 出生地', [('伦勃朗 - 维基百科', 'https://zh.wikipedia.org/wiki/伦勃朗', '伦勃朗出生于荷兰莱顿。')]),
         ('https://zh.wikipedia.org/wiki/莱顿', '莱顿位于哪条河', '莱顿位于旧莱茵河畔。')),
    # the teacher answered wrong: not admitted
    _row('Who wrote the novel the film Blade Runner adapts?', 'Philip K. Dick', 'Ridley Scott',
         ('Blade Runner novel', [('Do Androids Dream of Electric Sheep?', 'https://example.org/dads',
                                  'A 1968 novel by Philip K. Dick.')]),
         ('https://example.org/dads', 'author', 'Philip K. Dick wrote it.')),
    # the gold is in no recorded output: not admitted
    _row('What is the elevation of the summit nearest the town in question?', '2,962 m', 'It is 2,962 m.',
         ('summit elevation', [('Town', 'https://example.org/town', 'A town in the Alps.')]),
         ('https://example.org/town', 'elevation', 'The town is in the Alps.')),
]


def test_corpus_from_recorded_outputs():
    corpus, calls, stats = web_research.item_corpus([ROWS[0]['messages']])
    urls = [r['url'] for r in corpus['results']]
    assert urls == sorted(urls) and 'https://en.wikipedia.org/wiki/Rembrandt' in urls
    assert corpus['pages']['https://en.wikipedia.org/wiki/Leiden'][0]['text'].startswith(': \n\nEvidence in page')
    assert calls['searches'] == [{'query': 'Night Watch painter birthplace',
                                  'urls': ['https://en.wikipedia.org/wiki/Rembrandt', 'https://en.wikipedia.org/wiki/Leiden']}]
    assert len(calls['visits']) == 1 and not stats.get('search_unsplit')
    assert web_research.answer_kind('旧莱茵河') == 'cjk-short' and web_research.answer_kind('1,234') == 'latin-number'


def test_cases_service_and_replay(tmp_path):
    source = tmp_path / 'rows.jsonl.zst'
    _write(source, ROWS)
    assets = tmp_path / 'assets'
    cases = web_research.cases(path=str(source), out_assets=str(assets), report=str(tmp_path / 'log.jsonl'),
                               replay=str(tmp_path / 'replay.jsonl'), held_check=False)
    log = [json.loads(line) for line in (tmp_path / 'log.jsonl').read_text().splitlines()]
    assert [e.get('drop') for e in log] == [None, None, 'teacher_wrong', 'gold_not_in_corpus']
    assert [c['semantics']['expected'] for c in cases] == ['Oude Rijn', '旧莱茵河']
    for case in cases:
        files = case['semantics']['files']
        assert set(files) == {'research.nl', 'research/investigate.nl'}
        assert 'Neuralese<' in files['research/investigate.nl'] and 'investigate(brief)' in files['research.nl']
        assert case['semantics']['services']['web'].count(str(web_research.ASSET_ROOT)) == 1
        assert case['curriculum']['external']['checks']['keyword_baseline_code']
    if not (RUNTIME / 'dist/native/external.js').exists() or not _node():
        pytest.skip('needs the built ts-host runtime and node')
    rows = tmp_path / 'cases.jsonl'
    rows.write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in cases))
    out = tmp_path / 'replay.json'
    env = {**os.environ, 'NATLANG_RL_WEB_ASSETS': str(assets)}
    subprocess.run([_node(), str(ROOT / 'scripts/rl_tasks/web_research_replay.mjs'), str(RUNTIME), str(rows),
                    str(tmp_path / 'replay.jsonl'), str(out)], check=True, capture_output=True, text=True, timeout=120,
                   env=env)
    report = json.loads(out.read_text())
    assert report['failed'] == [] and report['summary']['visits_reproduced'] == 2
    probe = tmp_path / 'probe.mjs'
    probe.write_text(
        "import fs from 'node:fs'; import { createRequire } from 'node:module';\n"
        f"const {{ externalModule }} = await import({json.dumps(str(RUNTIME))} + '/dist/native/external.js');\n"
        f"const req = createRequire({json.dumps(str(RUNTIME))} + '/dist/native/external.js');\n"
        "const cases = fs.readFileSync(process.argv[2], 'utf8').split('\\n').filter(Boolean).map(JSON.parse);\n"
        "const out = cases.map(c => { const web = externalModule('web', c.semantics.services.web, req).exports;\n"
        "  return { en: web.search('Rembrandt born Leiden').map(r => r.url), zh: web.search('伦勃朗出生').map(r => r.url),\n"
        "    missing: web.visit('https://nowhere.example/', 'x'), other_goal: web.visit('https://en.wikipedia.org/wiki/Leiden', 'river') }; });\n"
        "console.log(JSON.stringify(out));\n")
    got = json.loads(subprocess.run([_node(), str(probe), str(rows)], check=True, capture_output=True, text=True,
                                    timeout=60, env=env).stdout)
    assert got[0]['en'][0] == 'https://en.wikipedia.org/wiki/Rembrandt'
    assert got[1]['zh'][0] == 'https://zh.wikipedia.org/wiki/伦勃朗'
    assert got[0]['missing'] == 'The page https://nowhere.example/ could not be opened.'
    assert 'Oude Rijn' in got[0]['other_goal']


def test_visit_header_names_the_callers_goal():
    corpus, calls, _ = web_research.item_corpus([ROWS[0]['messages']])
    page = corpus['pages']['https://en.wikipedia.org/wiki/Leiden'][0]
    assert page['head'] and 'which river' not in page['text']     # the teacher's goal is not stored in the reading
    assert calls['visits'][0]['output'] == ROWS[0]['messages'][5]['content'].split('<tool_response>\n')[1].split('\n</tool_response>')[0]
