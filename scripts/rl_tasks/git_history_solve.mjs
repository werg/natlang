#!/usr/bin/env node
// Reference check of the git-history source (scripts/rl_tasks/sources/git_history.py): solve every case through its
// own `git` service only (the model's view, compiled by the runtime's externalModule) and compare with the gold that
// git computed on the bare repository. A case whose service-derived answer differs, or that the service cannot answer,
// is reported and must be dropped.
// usage: node scripts/rl_tasks/git_history_solve.mjs RUNTIME_TS_HOST IN.jsonl OUT.json
import fs from 'node:fs';
import { createRequire } from 'node:module';
const [runtime, input, output] = process.argv.slice(2);
const { externalModule } = await import(runtime + '/dist/native/external.js');
const requireModule = createRequire(runtime + '/dist/native/external.js');

function bySubject(git, subject) {
  const hits = git.log().filter(c => c.subject === subject);
  if (hits.length !== 1) throw new Error(`${hits.length} commits with subject ${subject}`);
  return hits[0];
}
function addedLines(patch) {
  return patch.split('\n').filter(l => l.startsWith('+') && !l.startsWith('+++')).map(l => l.slice(1).trim());
}
function rewrites(patch) {
  const out = new Map(); let run = [];
  const close = () => { if (run.length === 2 && run[0][0] === '-' && run[1][0] === '+') out.set(run[1].slice(1).trim(), run[0].slice(1).trim()); run = []; };
  for (const line of patch.split('\n')) {
    if (line.startsWith('diff --git ') || line.startsWith('@@')) { close(); continue; }
    if (line.startsWith('+++') || line.startsWith('---') || line.startsWith('\\')) continue;
    if (line.startsWith('+') || line.startsWith('-')) run.push(line); else close();
  }
  close();
  return out;
}
function originOf(git, path, line) {
  const hits = git.log({ path }).filter(c => addedLines(git.diff(c.sha, path)).includes(line));
  if (hits.length !== 1) throw new Error(`${hits.length} commits add the line`);
  return hits[0].sha;
}
const solve = {
  line_origin: (git, p) => originOf(git, p.path, p.line),
  line_before: (git, p) => {
    const old = rewrites(git.diff(originOf(git, p.path, p.line), p.path)).get(p.line);
    if (old === undefined) throw new Error('no single-line rewrite');
    return old;
  },
  commit_files: (git, p) => git.show(bySubject(git, p.subject).sha).files.map(f => f.path).sort(),
  file_created: (git, p) => {
    const hits = git.log({ path: p.path }).filter(c => git.show(c.sha).files.some(f => f.path === p.path && f.status === 'A'));
    if (hits.length !== 1) throw new Error(`${hits.length} creating commits`);
    return hits[0].sha;
  },
  last_change_author: (git, p) => git.log({ path: p.path, max: 1 })[0].author,
  last_change_date: (git, p) => git.log({ path: p.path, max: 1 })[0].date.slice(0, 10),
  count_commits: (git, p) => {
    const all = git.log(p.dir ? { path: p.dir } : {});
    return all.filter(c => (p.author === undefined || c.author === p.author)
      && (p.since === undefined || c.date.slice(0, 10) >= p.since && c.date.slice(0, 10) <= p.until)
      && (p.prefix === undefined || c.subject.startsWith(p.prefix))
      && (p.status === undefined || git.show(c.sha).files.some(f => f.status === p.status))).length;
  },
  churn: (git, p) => {
    let added = 0, deleted = 0;
    for (const c of git.log({ path: p.path })) for (const f of git.show(c.sha).files) if (f.path === p.path) { added += f.added; deleted += f.deleted; }
    return { added, deleted };
  },
  tree_delta_removed: (git, p) => { const now = new Set(git.ls('HEAD')); return git.ls(bySubject(git, p.subject).sha).filter(x => !now.has(x)).sort(); },
  tree_delta_added: (git, p) => { const then = new Set(git.ls(bySubject(git, p.subject).sha)); return git.ls('HEAD').filter(x => !then.has(x)).sort(); },
  json_value_at: (git, p) => {
    let value = JSON.parse(git.cat(bySubject(git, p.subject).sha, p.path));
    for (const key of p.keys) value = value[key];
    return value;
  },
};
solve.signature_before = solve.line_before;
const canon = v => JSON.stringify(Array.isArray(v) ? [...v].sort() : v && typeof v === 'object' ? Object.fromEntries(Object.entries(v).sort()) : v);
const rows = fs.readFileSync(input, 'utf8').split('\n').filter(Boolean).map(JSON.parse);
const report = { schema: 'natlang.git-history-reference-check/1', cases: rows.length, agreed: 0, failed: [] };
const modules = new Map();
for (const row of rows) {
  const meta = row.curriculum.external.git;
  const source = row.semantics.services.git;
  if (!modules.has(source)) modules.set(source, externalModule('git', source, requireModule).exports);
  try {
    const got = solve[meta.kind](modules.get(source), meta.params);
    if (canon(got) === canon(row.semantics.expected)) report.agreed++;
    else report.failed.push({ id: row.id, kind: meta.kind, got, expected: row.semantics.expected });
  } catch (error) {
    report.failed.push({ id: row.id, kind: meta.kind, error: String(error.message ?? error) });
  }
}
report.failed_ids = report.failed.map(f => f.id);
fs.writeFileSync(output, JSON.stringify(report, null, 1) + '\n');
console.log(JSON.stringify({ cases: report.cases, agreed: report.agreed, failed: report.failed.length }));
