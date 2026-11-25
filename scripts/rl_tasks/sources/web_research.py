"""Web-research tasks from SearchSwarm (SearchSwarm/SearchSwarm-SFT, D02 of plans/neuralese/SUBAGENT_NEURALESE_20261124.md).

Each item is one real research question with a gold answer. The model answers it with a natlang program: the root
`research(question): string` and a callable-folder child `research/investigate(brief): Neuralese<Report>` that the root
may dispatch (the dataset's lead fans briefs out to sub-agents in the same way). Both reach the world through a crisp
`web` service over the item's own frozen corpus:

  - `web.search(query)`: BM25 over every search result the item's recorded trajectory saw (lead and kept sub-agents;
    title, url, snippet, plus the text of the page readings recorded for that url), top 10;
  - `web.visit(url, goal)`: the recorded reading of that url (the dataset's visit tool returns a goal-conditioned
    extraction, so a visit with another goal gets the recorded reading whose goal is closest: the main fidelity limit);
    an unrecorded url cannot be opened (its snippet is returned when a search showed it).

The corpus is one JSON asset per item bound by SHA-256 (`NATLANG_RL_WEB_ASSETS` overrides the root). No text is
invented: every result and reading is a recorded tool output of the item's trajectory.

Admission (per item): the gold answer (`answer`) is in the teacher's final answer (the item is answerable), it occurs in
the frozen corpus, and `web_research_replay.mjs` replays the teacher's own calls through the service: every recorded
visit reproduces its recorded output, and the gold is reachable through those calls. One task per distinct normalized
question. Oracle: the extractive-QA span oracle (token F1 >= 0.8 with containment, numbers exact; a Han/kana/hangul run
is one token, so a CJK answer must match its run exactly). Negative controls: other items' gold and keyword baselines
that answer from the top search result without any natural-language call.
"""
from __future__ import annotations

import collections
import hashlib
import json
import re
from pathlib import Path

from ..common import distinct_wrong, make_case, resolve_str, text_digest

SOURCE = 'searchswarm'
UPSTREAM = 'SearchSwarm/SearchSwarm-SFT'
REVISION = '4bad3397'
LICENSE = 'MIT'
INPUT = resolve_str('data_nvme', 'sa2nl-20261124', 'source', 'D02_train.parquet')
ASSET_ROOT = resolve_str('data_nvme', 'natlang-rl-web-research-20261125', 'assets')
SERVICE_TS = r'''import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
const FILE = __FILE__;
const SHA256 = __SHA__;
type Result = { title: string; url: string; snippet: string };
type Page = { goal: string; text: string; head: boolean };
type Corpus = { results: Result[]; pages: Record<string, Page[]> };
type Index = { corpus: Corpus; docs: Map<string, number>[]; lengths: number[]; df: Map<string, number>; mean: number;
  byUrl: Map<string, number> };
function tokens(text: string): string[] {
  const out: string[] = [];
  for (const run of String(text ?? '').normalize('NFKC').toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? []) {
    // Han, kana and hangul are written without spaces: index them as character bigrams.
    const parts = run.split(/([\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]+)/u);
    for (const part of parts) {
      if (!part) continue;
      if (/[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/u.test(part)) {
        if (part.length === 1) out.push(part);
        for (let k = 0; k + 1 < part.length; k++) out.push(part.slice(k, k + 2));
      } else out.push(part);
    }
  }
  return out;
}
function load(): Index {
  const cache = ((globalThis as any).__natlangRlWebResearch ??= new Map());
  if (!cache.has(SHA256)) {
    const bytes = readFileSync((process.env.NATLANG_RL_WEB_ASSETS ?? __ROOT__) + '/' + FILE);
    if (createHash('sha256').update(bytes).digest('hex') !== SHA256) throw new Error('web corpus asset changed: ' + FILE);
    const corpus: Corpus = JSON.parse(bytes.toString('utf8'));
    const docs: Map<string, number>[] = [], lengths: number[] = [], df = new Map<string, number>(), byUrl = new Map<string, number>();
    corpus.results.forEach((r, k) => {
      byUrl.set(r.url, k);
      const pages = (corpus.pages[r.url] ?? []).map(p => p.text).join(' ');
      const words = tokens(`${r.title} ${r.title} ${r.snippet} ${pages}`);
      const counts = new Map<string, number>();
      for (const w of words) counts.set(w, (counts.get(w) ?? 0) + 1);
      for (const w of counts.keys()) df.set(w, (df.get(w) ?? 0) + 1);
      docs.push(counts); lengths.push(words.length);
    });
    const mean = lengths.reduce((a, b) => a + b, 0) / Math.max(1, lengths.length);
    // A collector process runs many episodes: keep the few corpora in use, not every corpus it has ever opened.
    while (cache.size >= 12) cache.delete(cache.keys().next().value);
    cache.set(SHA256, { corpus, docs, lengths, df, mean, byUrl });
  }
  const index = cache.get(SHA256);
  cache.delete(SHA256); cache.set(SHA256, index);
  return index;
}
function rank(query: string, among: number[] | null, limit: number): number[] {
  const index = load(), n = index.docs.length, terms = [...new Set(tokens(query))];
  const scored: [number, number][] = [];
  for (const k of among ?? index.docs.keys()) {
    let score = 0;
    for (const t of terms) {
      const f = index.docs[k]!.get(t);
      if (!f) continue;
      const df = index.df.get(t)!;
      score += Math.log(1 + (n - df + 0.5) / (df + 0.5)) * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * index.lengths[k]! / index.mean));
    }
    if (score > 0) scored.push([score, k]);
  }
  return scored.sort((a, b) => b[0] - a[0] || a[1] - b[1]).slice(0, limit).map(([, k]) => k);
}
/** Searches the web for query and returns up to 10 results, best match first. Each result has the page's title, its url
 * and a snippet of its text; visit(url, goal) reads a page. One query per call: search several phrasings separately. */
export function search(query: string): Result[] {
  if (typeof query !== 'string' || !query.trim()) throw new Error('search(query) takes one non-empty query string');
  const corpus = load().corpus;
  return rank(query, null, 10).map(k => ({ ...corpus.results[k]! }));
}
/** Reads the page at url (a url from search results) and returns what it says that bears on goal, a sentence saying
 * what you are looking for on that page. */
export function visit(url: string, goal: string): string {
  if (typeof url !== 'string' || !url.trim()) throw new Error('visit(url, goal) takes the page url as a string');
  const index = load(), key = url.trim();
  const pages = index.corpus.pages[key] ?? index.corpus.pages[key.replace(/\/$/, '')] ?? index.corpus.pages[key + '/'];
  if (!pages?.length) {
    const k = index.byUrl.get(key);
    const snippet = k === undefined ? '' : index.corpus.results[k]!.snippet;
    return snippet ? `The page ${key} could not be opened. Its search snippet reads: ${snippet}`
      : `The page ${key} could not be opened.`;
  }
  // A page was read for particular goals: the reading whose goal is closest to this one.
  let best = 0, bestScore = -1;
  const want = new Set(tokens(String(goal ?? '')));
  pages.forEach((p, k) => {
    const have = tokens(p.goal);
    const score = p.goal === goal ? Infinity : have.filter(w => want.has(w)).length / Math.max(1, Math.sqrt(have.length));
    if (score > bestScore) { best = k; bestScore = score; }
  });
  const page = pages[best]!;
  // A recorded reading starts with a header naming the goal it was read for: it names the caller's goal instead.
  return page.head ? `The useful information in ${key} for user goal ${String(goal ?? "")} as follows${page.text}` : page.text;
}
'''

TOOL_CALL = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.S)
TOOL_RESPONSE = re.compile(r'<tool_response>\n?(.*?)\n?</tool_response>', re.S)
ANSWER = re.compile(r'<answer>\s*(.*?)\s*</answer>', re.S)
THINK = re.compile(r'<think>.*?</think>', re.S)
CJK = re.compile(r'[぀-ヿ㐀-鿿가-힯]')

ROOT_ARGS = {'question': 'string'}
REPORT = '{ findings: string; sources: { url: string; title: string }[]; open: string }'
INSTRUCTION = (
    'Answer question by researching it on the web. The question usually chains several facts: find each one in turn '
    '(an entity, then its date, then a place), since a later step depends on an earlier one. web.search(query) finds '
    'pages and returns their titles, urls and snippets; web.visit(url, goal) reads one page for goal, what you want to '
    'learn from it.\n\n'
    'Return only the final answer: a short name, number, date or phrase, written in the language and form the '
    'question asks for, not a sentence.')
ROOT_NOTE = (
    'investigate(brief) hands one self-contained sub-question to a helper that researches it on the web the same way '
    'and reports privately what it found, the pages it relied on and what stayed open. Give a brief everything the '
    'helper needs: it sees only the brief. Send independent sub-questions to investigate together, combine the reports, '
    'and check a deciding fact yourself when reports disagree or leave it open.')
HELPER = 'investigate'
HELPER_INSTRUCTION = (
    'Research brief on the web with web.search(query) and web.visit(url, goal) until you can answer it or the '
    'promising leads are used up. Search several phrasings, read the pages most likely to hold the facts, and follow a '
    'lead from one page to the next.\n\n'
    'Report findings: the facts that answer brief, each with the page it came from; sources: the url and title of each '
    'page you relied on; open: what you could not establish, or an empty string when nothing is open.')
ORACLE = {'level': 'span', 'normalization': 'qa', 'threshold': 0.8}


def qa_norm(text):
    text = (text or '').lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    return ' '.join(re.findall(r'\w+', text))


def parse_calls(content):
    """[{name, arguments}] of a recorded assistant reply (its <think> block ignored); None when a call does not parse."""
    out = []
    for raw in TOOL_CALL.findall(THINK.sub('', content or '')):
        try:
            call = json.loads(raw)
        except ValueError:
            return None
        if not isinstance(call, dict) or not isinstance(call.get('name'), str) or not isinstance(call.get('arguments'), dict):
            return None
        out.append(call)
    return out


def paired_calls(messages):
    """[(call, response text)] of one recorded conversation: each assistant reply's tool calls with the tool responses
    of the next user message, in order. Calls without a recorded response are left out."""
    out = []
    for i, message in enumerate(messages):
        if message['role'] != 'assistant':
            continue
        calls = parse_calls(message['content'])
        if not calls or i + 1 >= len(messages):
            continue
        following = messages[i + 1]
        bodies = TOOL_RESPONSE.findall(following['content'] or '') if following['role'] in ('user', 'tool') else []
        if following['role'] == 'tool' and not bodies:
            bodies = [following['content']]
        out += list(zip(calls, bodies))
    return out


def as_list(value):
    if value is None:
        return []
    return [v for v in (value if isinstance(value, list) else [value]) if isinstance(v, str) and v.strip()]


# ---- recorded tool outputs -> the item's corpus ------------------------------------------------------------------
SEARCH_TOOLS = ('search', 'google_scholar')


def split_outputs(text, n):
    """The per-argument parts of a batched tool output (one per query or url), or None when they cannot be told apart."""
    if n == 1:
        return [text]
    parts = re.split(r'\n=======\n', text)
    return parts if len(parts) == n else None


SEARCH_ENTRY = re.compile(r'(?m)^\d+\. \[(?P<title>.*?)\]\((?P<url>[^)\s]+)\)\n(?P<body>.*?)(?=^\d+\. \[|\Z)', re.S)


def search_entries(text):
    """[{title, url, snippet}] of one recorded search output (one query)."""
    out = []
    for m in SEARCH_ENTRY.finditer(text):
        body = m.group('body').strip()
        snippet = '\n'.join(line for line in body.splitlines()
                            if not re.match(r'^(Date published|Source|Authors?|Year|Cited by|Publication)\s*:', line)).strip()
        out.append({'title': m.group('title').strip(), 'url': m.group('url').strip(), 'snippet': snippet})
    return out


VISIT_HEAD = re.compile(r'The useful information in (?P<url>\S+?) for user goal (?P<goal>.*?) as follows', re.S)


def visit_output(url, goal, page):
    """What web.visit(url, goal) returns for a recorded reading (the service's rule, mirrored for the replay record)."""
    return f'The useful information in {url} for user goal {goal} as follows' + page['text'] if page['head'] else page['text']


def item_corpus(conversations):
    """(corpus, replay) of one item: every recorded search result and page reading of its conversations, and the
    recorded calls (searches with their result urls, visits with their exact outputs) for the reference replay."""
    results, pages, searches, visits, stats = {}, collections.defaultdict(list), [], [], collections.Counter()
    for messages in conversations:
        for call, body in paired_calls(messages):
            name, args = call['name'], call['arguments']
            if name in SEARCH_TOOLS:
                queries = as_list(args.get('query'))
                parts = split_outputs(body, len(queries)) if queries else None
                if parts is None:
                    stats['search_unsplit'] += 1
                    continue
                for query, part in zip(queries, parts):
                    entries = search_entries(part)
                    stats['search_entries'] += len(entries)
                    for entry in entries:
                        known = results.get(entry['url'])
                        if known is None or len(entry['snippet']) > len(known['snippet']):
                            results[entry['url']] = entry
                    searches.append({'query': query, 'urls': [e['url'] for e in entries]})
            elif name == 'visit':
                urls, goal = as_list(args.get('url')), args.get('goal') if isinstance(args.get('goal'), str) else ''
                parts = split_outputs(body, len(urls)) if urls else None
                if parts is None:
                    stats['visit_unsplit'] += 1
                    continue
                for url, part in zip(urls, parts):
                    if not part.strip():
                        continue
                    # The reading's header names the goal it was read for; the service writes the caller's own goal
                    # there, so a teacher's goal (often a guessed answer) never reaches the model.
                    head = f'The useful information in {url} for user goal {goal} as follows'
                    page = ({'goal': goal, 'text': part[len(head):], 'head': True} if part.startswith(head) else
                            {'goal': goal, 'text': part, 'head': False})
                    if not any(p['goal'] == goal for p in pages[url]):
                        pages[url].append(page)
                    visits.append({'url': url, 'goal': goal, 'output': visit_output(
                        url, goal, next(p for p in pages[url] if p['goal'] == goal))})
            else:
                stats[f'other:{name}'] += 1
    for url in pages:
        if url not in results:    # a page read without a recorded search result for it: listed by its url only
            results[url] = {'title': url, 'url': url, 'snippet': ''}
    corpus = {'results': sorted(results.values(), key=lambda r: r['url']),
              'pages': {url: pages[url] for url in sorted(pages)}}
    return corpus, {'searches': searches, 'visits': visits}, stats


# ---- items -> cases ------------------------------------------------------------------------------------------------
INPUT_ROWS = resolve_str('data_nvme', 'natlang-rl-web-research-20261125', 'source', 'D02_train.jsonl.zst')


def source_rows(path=INPUT_ROWS):
    """The release's rows, one JSON object per line (the parquet streamed to zstd JSONL once: its row groups are too
    large to decode in a small claim)."""
    import io
    import zstandard
    with open(path, 'rb') as raw:
        for line in io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(raw), encoding='utf-8'):
            if line.strip():
                yield json.loads(line)


def final_answer(messages):
    """The lead's final answer: the last assistant reply's <answer>, else its text without the think block."""
    for message in reversed(messages):
        if message['role'] == 'assistant':
            if parse_calls(message['content']):
                return None
            text = THINK.sub('', message['content'] or '')
            found = ANSWER.findall(text)
            return found[-1] if found else text.strip()
    return None


def answer_kind(answer):
    words = len(answer.split())
    script = 'cjk' if CJK.search(answer) else 'latin'
    kind = 'number' if re.fullmatch(r'[-+]?[\d.,%]+\s*\w{0,12}', answer.strip()) else 'short' if words <= 6 else 'phrase'
    return f'{script}-{kind}'


def service_source(rel, sha, root=ASSET_ROOT):
    return (SERVICE_TS.replace('__FILE__', json.dumps(rel)).replace('__SHA__', json.dumps(sha))
            .replace('__ROOT__', json.dumps(root)))


def write_asset(out_assets, rel, value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    target = Path(out_assets) / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or target.read_bytes() != data:
        tmp = target.with_suffix('.tmp')
        tmp.write_bytes(data)
        tmp.replace(target)
    return hashlib.sha256(data).hexdigest()


def corpus_text(corpus):
    return qa_norm(' '.join([*(f"{r['title']} {r['snippet']}" for r in corpus['results']),
                             *(p['text'] for pages in corpus['pages'].values() for p in pages)]))


KEYWORD_BASELINES = [
    # answer from the search engine alone, with no reading and no natural-language call
    'const hits = web.search(question);\nreturn hits.length ? hits[0].title : "";',
    'const hits = web.search(question);\nreturn hits.length ? hits[0].snippet.split(/[.。]/)[0] : "";',
]


def cases(path=INPUT_ROWS, out_assets=ASSET_ROOT, report=None, replay=None, limit=None, seed=20261125,
          held_check=True):
    """Cases of every admissible item (module docstring). ``report``: per-item admission log (JSONL); ``replay``: the
    recorded calls per case for web_research_replay.mjs (JSONL)."""
    import random
    held = None
    if held_check:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts' / 'neuralese_data'))
        from subagent_neuralese import held_content_hit, held_texts
        held = held_texts()
    empty_sha = write_asset(out_assets, 'web-research/empty.json', {'results': [], 'pages': {}})
    empty_service = service_source('web-research/empty.json', empty_sha, str(out_assets))
    log, items, seen = [], [], set()
    # Replay records go straight to disk: the recorded visit outputs of every item do not fit in memory together.
    replay_handle = open(replay, 'w') if replay else None
    for index, row in enumerate(source_rows(path)):
        if limit is not None and index >= limit:
            break
        question, gold = (row.get('question') or '').strip(), (row.get('answer') or '').strip()
        key = hashlib.sha256(question.encode()).hexdigest()[:16]
        entry = {'index': index, 'key': key, 'source': row.get('source')}
        log.append(entry)
        final = final_answer(row['messages'])
        if not question or not qa_norm(gold):
            entry['drop'] = 'no_question_or_gold'
            continue
        if not final or qa_norm(gold) not in qa_norm(final):
            entry['drop'] = 'teacher_wrong'
            continue
        if qa_norm(question) in seen:
            entry['drop'] = 'duplicate_question'
            continue
        if qa_norm(gold) in qa_norm(question):
            entry['drop'] = 'gold_in_question'
            continue
        if held is not None and held_content_hit(question, held):
            entry['drop'] = 'held_content'
            continue
        subagents = row.get('subagents') or []
        corpus, calls, stats = item_corpus([row['messages'], *[s['messages'] for s in subagents]])
        entry.update({'results': len(corpus['results']), 'pages': len(corpus['pages']), **stats})
        if qa_norm(gold) not in corpus_text(corpus):
            entry['drop'] = 'gold_not_in_corpus'
            continue
        seen.add(qa_norm(question))
        rel = f'web-research/{row.get("source")}-{key}.json'
        sha = write_asset(out_assets, rel, corpus)
        dispatched = sum(len(c['arguments'].get('prompts') or ()) for m in row['messages'] if m['role'] == 'assistant'
                         for c in parse_calls(m['content']) or () if c['name'] == 'call_sub_agent')
        entry['admitted'] = True
        upstream_id = f'{row.get("source")}:{key}'
        if replay_handle:
            replay_handle.write(json.dumps({'upstream_id': upstream_id, 'gold': gold, **calls,
                                            'empty_service': empty_service}, ensure_ascii=False) + '\n')
        items.append({'row_source': row.get('source'), 'upstream_id': upstream_id, 'index': index, 'key': key,
                      'question': question, 'gold': gold, 'kind': answer_kind(gold), 'rel': rel, 'sha': sha,
                      'dispatched': dispatched, 'kept_subagents': len(subagents),
                      'teacher_calls': len(calls['searches']) + len(calls['visits'])})
    pools = collections.defaultdict(list)
    for item in items:
        pools[item['kind']].append(item['gold'])
    for pool in pools.values():
        random.Random(seed).shuffle(pool)
    if replay_handle:
        replay_handle.close()
    out = []
    for item in items:
        gold, kind = item['gold'], item['kind']
        wrong = distinct_wrong(gold, [g for g in pools[kind] if qa_norm(g) != qa_norm(gold)], 2)
        aliases = [gold + '。', f'「{gold}」'] if kind.startswith('cjk') else [gold.upper(), gold + '.', f'"{gold}"']
        level = 3 if item['teacher_calls'] > 30 or item['dispatched'] > 6 else 2
        case = make_case(
            source=SOURCE, family=f'web_research_{kind.replace("-", "_")}', shape=f'web-research-{kind}',
            upstream=UPSTREAM, upstream_id=item['upstream_id'], upstream_split='train-carve',
            unit=f'q:{item["key"]}', groups=[f'searchswarm:{item["key"]}'], root_name='research', args=ROOT_ARGS,
            returns='string', instruction=INSTRUCTION, inputs={'question': item['question']}, expected=gold,
            oracle=dict(ORACLE), level=level, license_=LICENSE, services={'web': service_source(item['rel'], item['sha'])},
            domain='relational', slice_='nested_scoped', wrong=wrong, aliases=aliases, revision=REVISION,
            neuralese={'helper': HELPER, 'args': {'brief': 'string'}, 'returns': f'Neuralese<{REPORT}>',
                       'instruction': HELPER_INSTRUCTION, 'root_note': ROOT_NOTE},
            framing='neuralese-helper',
            neuralese_room='sub-questions researched by investigate children whose reports reach the parent as Neuralese')
        external = case['curriculum']['external']
        external['checks']['keyword_baseline_code'] = KEYWORD_BASELINES
        external['web'] = {'asset': item['rel'], 'asset_sha256': item['sha'], 'upstream_row': item['index'],
                           'dataset_source': item['row_source'], 'answer_kind': kind,
                           'teacher_calls': item['teacher_calls'], 'teacher_dispatched': item['dispatched'],
                           'teacher_subagents_kept': item['kept_subagents']}
        out.append(case)
    if report:
        Path(report).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in log))
    return out
