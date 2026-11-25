#!/usr/bin/env node
// Reference replay of the web-research source (scripts/rl_tasks/sources/web_research.py): replay every recorded call of
// the item's trajectory through the case's own `web` service (the model's view, compiled by the runtime's
// externalModule).
//   - every recorded visit(url, goal) must return its recorded reading exactly;
//   - a recorded search(query) is measured by how many of its recorded result urls come back in the top 10 (recall; not
//     a keep condition, since our ranking is BM25 over the item's own corpus, not the dataset's search engine);
//   - the gold must be reachable: some replayed visit output or replayed search snippet contains it (QA-normalized).
// Negative control: the same calls on an empty corpus return no result and open no page.
// usage: node scripts/rl_tasks/web_research_replay.mjs RUNTIME_TS_HOST CASES.jsonl REPLAY.jsonl OUT.json
import fs from 'node:fs';
import { createRequire } from 'node:module';
const [runtime, casesPath, replayPath, output] = process.argv.slice(2);
const { externalModule } = await import(runtime + '/dist/native/external.js');
const requireModule = createRequire(runtime + '/dist/native/external.js');
const read = p => fs.readFileSync(p, 'utf8').split('\n').filter(Boolean).map(JSON.parse);
// Replay records hold every recorded visit output: stream them (the cases fit in memory, the records do not).
const cases = new Map(read(casesPath).map(r => [r.curriculum.external.upstream_id, r]));
const norm = s => String(s ?? '').normalize('NFKC').toLowerCase().replace(/\b(a|an|the)\b/g, ' ')
  .match(/[\p{L}\p{N}]+/gu)?.join(' ') ?? '';
const results = [];
const seen = new Set();
// Lines end at \n only: readline also breaks at U+2028/U+2029, which recorded page text holds verbatim.
async function* lines(path) {
  let rest = '';
  for await (const chunk of fs.createReadStream(path, { encoding: 'utf8' })) {
    const parts = (rest + chunk).split('\n');
    rest = parts.pop();
    yield* parts;
  }
  if (rest) yield rest;
}
for await (const line of lines(replayPath)) {
  if (!line.trim()) continue;
  const replay = JSON.parse(line), record = cases.get(replay.upstream_id);
  if (!record) continue;
  seen.add(replay.upstream_id);
  const result = { id: record.id, visits: 0, visits_reproduced: 0, searches: 0, search_recall: null, gold_reachable: false,
    empty_corpus_silent: null, problems: [] };
  try {
    const web = externalModule('web', record.semantics.services.web, requireModule).exports;
    const gold = norm(replay.gold);
    let found = 0, wanted = 0;
    for (const call of replay.visits) {
      result.visits++;
      const got = web.visit(call.url, call.goal);
      if (got === call.output) result.visits_reproduced++;
      else if (result.problems.length < 5) result.problems.push(`visit ${call.url} differs`);
      if (gold && norm(got).includes(gold)) result.gold_reachable = true;
    }
    for (const call of replay.searches) {
      result.searches++;
      const got = web.search(call.query);
      const urls = new Set(got.map(r => r.url));
      wanted += call.urls.length; found += call.urls.filter(u => urls.has(u)).length;
      if (gold && got.some(r => norm(`${r.title} ${r.snippet}`).includes(gold))) result.gold_reachable = true;
    }
    result.search_recall = wanted ? Number((found / wanted).toFixed(4)) : null;
    // Negative control: the same service source over an empty corpus.
    const empty = replay.empty_service ? externalModule('web', replay.empty_service, requireModule).exports : null;
    if (empty) {
      result.empty_corpus_silent = replay.searches.every(c => empty.search(c.query).length === 0) &&
        replay.visits.every(c => /could not be opened/.test(empty.visit(c.url, c.goal)));
    }
  } catch (e) { result.problems.push(String(e).slice(0, 300)); }
  result.ok = result.problems.length === 0 && result.visits_reproduced === result.visits && result.gold_reachable &&
    result.empty_corpus_silent !== false;
  results.push(result);
  if (results.length % 200 === 0) console.error(JSON.stringify({ done: results.length, of: cases.size }));
}
for (const [upstream, record] of cases) if (!seen.has(upstream))
  results.push({ id: record.id, visits: 0, visits_reproduced: 0, searches: 0, search_recall: null, gold_reachable: false,
    empty_corpus_silent: null, problems: ['no replay record'], ok: false });
const ok = results.filter(r => r.ok);
const recall = results.map(r => r.search_recall).filter(x => x !== null);
const summary = { cases: results.length, ok: ok.length,
  visits: results.reduce((n, r) => n + r.visits, 0), visits_reproduced: results.reduce((n, r) => n + r.visits_reproduced, 0),
  gold_unreachable: results.filter(r => !r.gold_reachable).length,
  empty_corpus_not_silent: results.filter(r => r.empty_corpus_silent === false).length,
  mean_search_recall: recall.length ? Number((recall.reduce((a, b) => a + b, 0) / recall.length).toFixed(4)) : null };
fs.writeFileSync(output, JSON.stringify({ schema: 'natlang.web-research-replay/1', summary,
  failed: results.filter(r => !r.ok).map(r => r.id), results }) + '\n');
console.log(JSON.stringify(summary));
