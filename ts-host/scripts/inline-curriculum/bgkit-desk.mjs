// bgkit's promptable compression as NatLang programs (owner 2026-10-06): a bgkit compression prompt ("Give me a compact
// digest of this result for: ...", "Compress the file to answer: ...") becomes the instructions of an inline nl lambda,
// and the consumer of the compressed text is a program step (an answering lambda, or the caller itself). Rows come from
// the bgkit stores exported by scripts/neuralese_data/export_bgkit_jsonl.py (data only; bgkit's harness stays behind).
//
// digest_desk: several agent tool results (bash, read, edit, search over real repositories), each with the need it was
//   run for; the program digests each with one lambda and returns the digests. The oracle checks code-side that the
//   digests keep the exact identifiers, paths and numbers of bgkit's gold digest that occur in the raw result, and that
//   the whole stays compact.
// repo_answer: encode then consume. One lambda quotes the lines of a source file that answer a question about it, a
//   second answers from the quotes alone. Questions and answers are code-derived (signatures, definitions, callers,
//   importers ...); the quoted lines are the file's lines that hold the answer.
import { readFileSync } from 'node:fs';
import { checkConstraints } from '../../dist/evaluation/constraints.js';
import { Random, curriculumCase, evalCall, literal, returnCall } from './lib.mjs';

const EXPORT = process.env.NATLANG_BGKIT_EXPORT ?? '/home/werg/data/bgkit-export';
const cache = new Map();
function store(name) {
  if (!cache.has(name)) {
    let text;
    try { text = readFileSync(`${EXPORT}/${name}.jsonl`, 'utf8'); }
    catch { throw new Error(`${name} is not exported; run .venv-neuralese/bin/python -m scripts.neuralese_data.export_bgkit_jsonl ${name}`); }
    cache.set(name, text.split('\n').filter(Boolean).map(line => JSON.parse(line)));
  }
  return cache.get(name);
}
// bgkit's eval split is our test split.
const inSplit = (row, split) => (row.split === 'eval' ? 'test' : 'train') === split;
const plainRun = (text, n = 60) => (/^[A-Za-z0-9 ,.'()_-]*/.exec(text.slice(0, n))?.[0] ?? '').trim();
// Source repositories carry no license metadata in bgkit, so the license is unreviewed and cases stay candidates until
// source and need completeness are reviewed (Pop review 2026-10-06); rows resolve through the pinned export snapshot.
const LICENSE = 'LicenseRef-unreviewed (bgkit task stores over public repositories; per-repository licenses not recorded)';
const EXPORT_SNAPSHOT = 'bgkit-exports-20261006-v1';
const candidate = record => Object.assign(record.generation, { admission: 'candidate-pending-source-review', source_snapshot: EXPORT_SNAPSHOT });

// Identifier-like tokens: paths, dotted names, snake/camel case, numbers of two or more digits.
const TOKEN = /[A-Za-z_][\w.\/-]*\w|\d{2,}/g;
const identifierLike = token => /[_./]|\d|[a-z][A-Z]/.test(token) && token.length >= 3;

let digestPool;
function digestRows(split) {
  digestPool ??= [...store('tool_digest_v2'), ...store('tool_digest_ext_v3')].filter(row => {
    if (!row.context || !row.target || row.context.length > 8000 || row.target.length > 1200) return false;
    // A gold digest that names things its raw result does not show (bgkit's synthetic digests sometimes do) is noise.
    const named = [...new Set((row.target.match(TOKEN) ?? []).filter(identifierLike))];
    const shown = named.filter(token => row.context.includes(token));
    return named.length >= 2 && shown.length / named.length >= 0.6;
  });
  return digestPool.filter(row => inSplit(row, split));
}

function anchors(row) {
  const shown = [...new Set((row.target.match(TOKEN) ?? []).filter(identifierLike))].filter(token => row.context.includes(token));
  return shown.sort((a, b) => b.length - a.length || (a < b ? -1 : 1)).slice(0, 4);
}

const needOf = row => /for: ([\s\S]*?)\.?\s*Keep every exact/.exec(row.instruction)?.[1]?.trim() ??
  /matches for '([^']+)'/.exec(row.instruction)?.[1] ?? row.instruction;

export function digestDesk(seed, index, split = 'train') {
  const rng = new Random(seed, `digest_desk:${index}`);
  const pool = digestRows(split);
  for (let attempt = 0; attempt < 50; attempt++) {
    const rows = rng.sample(pool, rng.int(3, 5));
    const results = rows.map(row => ({ tool: row.tool_name, args: row.tool_args, need: needOf(row), output: row.context }));
    const markers = results.map(result => plainRun(result.need));
    if (markers.some(marker => marker.length < 16) || new Set(markers).size !== markers.length) continue;
    const parts = rows.map((row, i) => `### ${results[i].tool} ${results[i].args}\n${row.target}`);
    const final = parts.join('\n\n');
    const goldWords = rows.reduce((n, row) => n + row.target.split(/\s+/).length, 0);
    const constraints = [{ kind: 'include_words', words: rows.flatMap(anchors) },
      { kind: 'word_count', max: Math.ceil(goldWords * 1.6) + 12 * rows.length }];
    if (!checkConstraints(final, constraints).passed) continue;
    const code = `const digest = nl<(output: string, need: string) => Promise<string>>\`Give a compact digest of output for need. Keep every exact identifier, path, line number and value that matters; drop the rest.\`;
const parts: string[] = [];
for (const result of session.results()) parts.push(\`### \${result.tool} \${result.args}\\n\${await digest(result.output, result.need)}\`);
return parts.join('\\n\\n');`;
    const record = curriculumCase({ family: 'digest_desk', shape: `digest${index}`, variant: 'v0', slice: 'inline_placement', domain: 'other',
      splitGroup: `bgkit-digest:${rows.map(row => `${row.store}:${row.index}`).sort().join(',')}`,
      mode: 'single_call', inline: 'required',
      evidence: { world: results.map(result => result.need), retrieved: [], background: [] }, decisive: [],
      minimumSequence: ['read the tool results and the need each was run for', 'digest each with an inline nl lambda given its need',
        'return the digests under their headers'],
      reference: { root: [evalCall(code), returnCall(final)],
        children: rows.map((row, i) => ({ match: ['Give a compact digest of output', markers[i]], value: row.target })) },
      root: { name: 'digest_results', args: {}, returns: 'string',
        instructions: 'session.results() holds tool results from an agent working in a repository, each with the need it was run for. Digest each result for its need with an inline nl lambda: keep every exact identifier, path, line number and value that matters and drop the rest. Return the digests in order, each under a header line "### <tool> <args>", separated by blank lines.' },
      files: { 'digest_results/session.ts': `const RESULTS: { tool: string, args: string, need: string, output: string }[] = ${literal(results)};\n` +
        '/** The tool results to digest, in order: the tool, its arguments, the need it was run for, and its raw output. */\n' +
        'export function results(): { tool: string, args: string, need: string, output: string }[] { return RESULTS; }\n' },
      inputs: {}, expected: constraints, split });
    record.semantics.oracle = { level: 'constraints' };
    record.license = LICENSE;
    record.gold_sources = ['bgkit-tool-digest-targets', 'code-checked-identifier-retention'];
    record.dataset = 'bgkit';
    record.dataset_records = rows.map(row => `${row.store}:${row.index}`);
    candidate(record);
    return [record];
  }
  return [];
}

let repoPool;
function repoRows(split) {
  repoPool ??= store('repo_qa_file_v3').filter(row => row.context?.startsWith('# file: ') && row.context.length <= 12000 &&
    row.target && row.target.length <= 200 && !row.target.includes('\n'));
  return repoPool.filter(row => inSplit(row, split));
}

/** The file's lines that hold the answer: lines naming the question's terms or the answer, with their enclosing
 * definitions, as "L<n>: text". Null when the quotes do not contain the answer. */
function quotes(lines, question, answer) {
  const terms = [...question.matchAll(/`([^`]+)`/g)].map(match => match[1].split('/').pop()).filter(term => term.length >= 2);
  const answerParts = answer.split(/,\s*/).map(part => part.trim()).filter(Boolean);
  const wanted = new Set();
  lines.forEach((line, i) => {
    if (![...terms, ...answerParts].some(term => line.includes(term))) return;
    wanted.add(i);
    for (let j = i - 1; j >= 0; j--) if (/^\s*(?:async\s+)?(?:def|class|function|func|fn)\s/.test(lines[j])) { wanted.add(j); break; }
  });
  const chosen = [...wanted].sort((a, b) => a - b).slice(0, 12);
  const text = chosen.map(i => `L${i + 1}: ${lines[i]}`).join('\n');
  return answerParts.every(part => text.includes(part)) ? text : null;
}

export function repoAnswer(seed, index, split = 'train') {
  const rng = new Random(seed, `repo_answer:${index}`);
  const pool = repoRows(split);
  for (let attempt = 0; attempt < 50; attempt++) {
    const row = rng.pick(pool);
    const [header, ...lines] = row.context.split('\n');
    const path = header.slice('# file: '.length).trim();
    const question = row.instruction.trim();
    const notes = quotes(lines, question, row.target);
    if (!notes || plainRun(question).length < 12) continue;
    const source = lines.join('\n');
    const code = `const quote = nl<(source: string, question: string) => Promise<string>>\`Quote the lines of source that answer question, each as "L<line number>: <line>". Quote exactly; add nothing.\`;
const answer = nl<(notes: string, question: string) => Promise<string>>\`Answer question from notes alone, exactly as the code writes it.\`;
const notes = await quote(repo.text(), question);
return await answer(notes, question);`;
    const record = curriculumCase({ family: 'repo_answer', shape: `repo${index}`, variant: 'v0', slice: 'inline_placement', domain: 'other',
      splitGroup: `bgkit-repo:${row.meta.repo ?? path}`, mode: 'single_call', inline: 'required',
      evidence: { world: [question], retrieved: [path], background: [] }, decisive: [],
      minimumSequence: ['quote the lines of the file that answer the question with an inline nl lambda',
        'answer from the quoted lines alone with a second lambda'],
      reference: { root: [evalCall(code), returnCall(row.target)],
        children: [{ match: ['Quote the lines of source that answer question', plainRun(question)], value: notes },
          { match: ['Answer question from notes alone'], value: row.target }] },
      root: { name: 'answer_from_file', args: { question: 'string' }, returns: 'string',
        instructions: `repo.text() is the source file ${path}; repo.path() is its path. Answer question about it in two inline nl steps: the first quotes the lines that answer question, each as "L<line number>: <line>"; the second answers from those quotes alone. Return the answer exactly as the code writes it (a signature, path, name or list of names).` },
      files: { 'answer_from_file/repo.ts': `const PATH = ${JSON.stringify(path)};\nconst TEXT = ${JSON.stringify(source)};\n` +
        '/** The path of the source file. */\nexport function path(): string { return PATH; }\n' +
        '/** The text of the source file. */\nexport function text(): string { return TEXT; }\n' },
      inputs: { question }, expected: row.target, split });
    record.semantics.oracle = { level: 'span', threshold: 0.8, normalization: 'qa',
      rubric: 'Answer the question about the source file. The answer must name the same code element(s) as the reference; for lists, the same set of names.',
      context: { question, quoted_lines: notes } };
    record.curriculum.answer_evidence = notes.split('\n');
    record.license = LICENSE;
    record.gold_sources = ['bgkit-repo-qa-code-derived-answers'];
    record.dataset = 'bgkit';
    record.dataset_records = [`${row.store}:${row.index}`];
    candidate(record);
    return [record];
  }
  return [];
}
