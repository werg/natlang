#!/usr/bin/env node
// Build small, source-derived task pools whose labels are checked by independent host solvers.
// The output is a task variant over existing S1 records; it creates no new source-world credit.
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { curriculumCase, evalCall, returnCall } from './lib.mjs';

const { values } = parseArgs({ options: { corpus: { type: 'string' }, out: { type: 'string' } } });
if (!values.corpus || !values.out) throw new Error('usage: build-verified-reasoning-pools-v1.mjs --corpus DIR --out NEW_DIR');
const corpus = resolve(values.corpus), out = resolve(values.out);
const sha = value => createHash('sha256').update(value).digest('hex');
const canonical = value => JSON.stringify(value);
const lines = async function* (path, hash) {
  let pending = Buffer.alloc(0);
  for await (const chunk of createReadStream(path)) {
    hash.update(chunk);
    pending = pending.length ? Buffer.concat([pending, chunk]) : chunk;
    let i;
    while ((i = pending.indexOf(0x0a)) >= 0) {
      const row = pending.subarray(0, i).toString('utf8');
      pending = pending.subarray(i + 1);
      if (row.trim()) yield JSON.parse(row);
    }
  }
  if (pending.length) yield JSON.parse(pending.toString('utf8'));
};

const selectedIds = {
  reasoning_gym: [0, 1, 2, 3, 4, 5, 6, 8, 9, 11, 14, 18, 19, 22, 23, 24]
    .map(index => `sdkb:reasoning-gym:reasoning_gym-${index}`),
  synlogic_candidates: [],
};
const synlogicSupported = new Set([
  'sdkb:synlogic:synlogic-easy-train-1', 'sdkb:synlogic:synlogic-easy-train-28',
  'sdkb:synlogic:synlogic-easy-train-153', 'sdkb:synlogic:synlogic-easy-train-207',
  'sdkb:synlogic:synlogic-easy-train-263', 'sdkb:synlogic:synlogic-easy-train-288',
  'sdkb:synlogic:synlogic-easy-train-308',
]);
// Candidate pool includes the seven demonstration-checkable grids and nine explicit omissions.
// The unsupported records are documented, not silently added to the generated cases.
selectedIds.synlogic_candidates = [
  'sdkb:synlogic:synlogic-easy-train-0', 'sdkb:synlogic:synlogic-easy-train-1',
  'sdkb:synlogic:synlogic-easy-train-2', 'sdkb:synlogic:synlogic-easy-train-3',
  'sdkb:synlogic:synlogic-easy-train-4', 'sdkb:synlogic:synlogic-easy-train-5',
  'sdkb:synlogic:synlogic-easy-train-6', 'sdkb:synlogic:synlogic-easy-train-8',
  'sdkb:synlogic:synlogic-easy-train-9', 'sdkb:synlogic:synlogic-easy-train-11',
  'sdkb:synlogic:synlogic-easy-train-28', 'sdkb:synlogic:synlogic-easy-train-153',
  'sdkb:synlogic:synlogic-easy-train-207', 'sdkb:synlogic:synlogic-easy-train-263',
  'sdkb:synlogic:synlogic-easy-train-288', 'sdkb:synlogic:synlogic-easy-train-308',
];
const familyFiles = {
  reasoning_gym: 'reasoning_gym.port-records.jsonl',
  reasoning_synlogic: 'reasoning_synlogic.port-records.jsonl',
};
const wanted = new Set([...selectedIds.reasoning_gym, ...selectedIds.synlogic_candidates]);
const sourceRecords = new Map();
const familyDigests = {};
const corpusManifestBytes = await readFile(resolve(corpus, 'manifest.json'));
for (const [family, filename] of Object.entries(familyFiles)) {
  const hash = createHash('sha256');
  let count = 0;
  for await (const row of lines(resolve(corpus, filename), hash)) {
    count++;
    if (wanted.has(row.id)) sourceRecords.set(row.id, row);
  }
  familyDigests[family] = { file: filename, sha256: hash.digest('hex'), record_count: count };
}
for (const id of wanted) {
  const row = sourceRecords.get(id);
  if (!row) throw new Error(`selected source id not found: ${id}`);
  const wantedFamily = id.startsWith('sdkb:reasoning-gym:') ? 'reasoning_gym' : 'reasoning_synlogic';
  if (row.family !== wantedFamily || row.split !== 'train' || row.license?.spdx !== 'Apache-2.0' && row.license?.spdx !== 'MIT')
    throw new Error(`${id}: family/split/license changed from reviewed source contract`);
}

const tokenPattern = /(?<![A-Za-z0-9])[A-Za-z][&@#$+]|[&@#$+][A-Za-z](?![A-Za-z0-9])/g;
function parseRewriteQuestion(question) {
  const tokens = [...new Set([...question.matchAll(/`([^`\s]+)`/g)].map(match => match[1])
    .filter(token => /^(?:[A-Za-z][&@#$+]|[&@#$+][A-Za-z])$/.test(token)))];
  if (tokens.length !== 4) throw new Error('rewrite question does not define four two-character tokens');
  const rules = new Map();
  for (const line of question.split('\n')) {
    const text = line.trim().replace(/^[-*]\s*/, '');
    const matches = [...text.matchAll(tokenPattern)].map(match => match[0]);
    if (matches.length < 2) continue;
    const [left, right] = matches;
    if (!tokens.includes(left) || !tokens.includes(right) || left[1] !== right[0]) continue;
    let replacement;
    if (/\b(?:is removed|removed|nothing|empty)\b/i.test(text)) replacement = [];
    else if (/\b(?:is replaced by|becomes)\b/i.test(text)) {
      const actionIndex = text.search(/\b(?:is replaced by|becomes)\b/i);
      const rhs = [...text.slice(actionIndex).matchAll(tokenPattern)].map(match => match[0]).filter(token => tokens.includes(token));
      if (rhs.length !== 2) continue;
      replacement = rhs;
    } else continue;
    const key = `${left}\u0000${right}`;
    if (rules.has(key) && canonical(rules.get(key)) !== canonical(replacement)) throw new Error(`conflicting rewrite rule ${left} ${right}`);
    rules.set(key, replacement);
  }
  if (rules.size !== 4) throw new Error(`expected exactly four directed rules, found ${rules.size}`);
  const markers = [...question.matchAll(/(?:following program:|program is:|program:)([^\n]*)/gi)];
  if (!markers.length) throw new Error('cannot locate the program in rewrite question');
  let tail = markers.at(-1)[1];
  if (!tail.trim()) tail = question.slice(markers.at(-1).index + markers.at(-1)[0].length).split('\n').find(line => line.trim()) ?? '';
  const program = tail.trim().replaceAll('`', '').replace(/[.]\s*$/, '').split(/\s+/).filter(Boolean);
  if (!program.length || program.some(token => !tokens.includes(token))) throw new Error('program contains unknown tokens');
  return { tokens, rules: Object.fromEntries(rules), program };
}
function solveRewrite(question) {
  const { rules, program } = parseRewriteQuestion(question);
  let state = [...program];
  const seen = new Set();
  for (let step = 0; step < 10000; step++) {
    const fingerprint = canonical(state);
    if (seen.has(fingerprint)) throw new Error('rewrite system entered a repeated state before reaching a terminal sequence');
    seen.add(fingerprint);
    let applied = false;
    for (let i = 0; i + 1 < state.length; i++) {
      const replacement = rules[`${state[i]}\u0000${state[i + 1]}`];
      if (replacement) { state = [...state.slice(0, i), ...replacement, ...state.slice(i + 2)]; applied = true; break; }
      if (Object.hasOwn(rules, `${state[i]}\u0000${state[i + 1]}`)) { state = [...state.slice(0, i), ...state.slice(i + 2)]; applied = true; break; }
    }
    if (!applied) return state.length ? state.join(' ') : 'empty';
  }
  throw new Error('rewrite system did not terminate within 10000 steps');
}
function targetFinalSequence(target) {
  const labels = [...target.matchAll(/(?:final state|final sequence|final program)\s*:?\s*/ig)];
  for (const match of labels.reverse()) {
    const tail = target.slice(match.index + match[0].length);
    const quoted = tail.match(/`([^`]+)`/);
    let text = quoted?.[1] ?? tail.split('\n')[0].replace(/\s*(?:<\/think>|\*\*).*/, '').trim();
    text = text.replace(/[.]$/, '').trim();
    if (/^empty$/i.test(text)) return 'empty';
    if (text && text.split(/\s+/).every(token => /^(?:[A-Za-z][&@#$+]|[&@#$+][A-Za-z])$/.test(token))) return text;
  }
  throw new Error('could not parse final token sequence from source answer');
}
function parseGridLabels(text, label) {
  const found = [];
  const pattern = new RegExp(`^\\s*${label}\\s*:`, 'gim');
  for (const match of text.matchAll(pattern)) {
    let start = match.index + match[0].length;
    while (/\s/.test(text[start] ?? '')) start++;
    if (text[start] !== '[') continue;
    let depth = 0, quoted = false, escape = false;
    for (let i = start; i < text.length; i++) {
      const char = text[i];
      if (quoted) { if (escape) escape = false; else if (char === '\\') escape = true; else if (char === '"') quoted = false; }
      else if (char === '"') quoted = true;
      else if (char === '[') depth++;
      else if (char === ']' && --depth === 0) { found.push(JSON.parse(text.slice(start, i + 1))); break; }
    }
  }
  return found;
}
const gridShape = grid => Array.isArray(grid) && grid.length > 0 && grid.every(row => Array.isArray(row) && row.length > 0 && row.every(value => Number.isInteger(value))) && new Set(grid.map(row => row.length)).size === 1;
function transforms(grid) {
  const out = [];
  let rotated = grid;
  for (let k = 1; k <= 4; k++) {
    rotated = rotated[0].map((_, column) => rotated.map(row => row[column]).reverse());
    out.push([`rotate-${k}`, rotated]);
    out.push([`rotate-${k}-mirror-horizontal`, rotated.map(row => [...row].reverse())]);
  }
  return out;
}
function checkSynlogic(row) {
  const question = row.consumer.context.find(message => message.role === 'user')?.content;
  if (!question) throw new Error(`${row.id}: missing user prompt`);
  const inputs = parseGridLabels(question, 'Input').filter(gridShape);
  const outputs = parseGridLabels(question, 'Output').filter(gridShape);
  const answerTag = row.target.value.match(/<answer>\s*(\[.*?\])\s*<\/answer>/s);
  if (!answerTag) throw new Error(`${row.id}: target does not contain a single JSON-grid answer tag`);
  const gold = JSON.parse(answerTag[1]);
  if (!gridShape(gold) || inputs.length !== outputs.length + 1 || outputs.length < 2)
    return { supported: false, reason: 'unsupported grid presentation or incomplete examples' };
  const predictions = [];
  for (const [name] of transforms(inputs[0])) {
    const transform = grid => transforms(grid).find(([candidate]) => candidate === name)[1];
    const map = new Map();
    let valid = true;
    for (let i = 0; i < outputs.length && valid; i++) {
      const transformed = transform(inputs[i]), expected = outputs[i];
      if (transformed.length !== expected.length || transformed.some((line, rowIndex) => line.length !== expected[rowIndex].length)) { valid = false; break; }
      for (let y = 0; y < transformed.length && valid; y++) for (let x = 0; x < transformed[y].length; x++) {
        const source = transformed[y][x], target = expected[y][x];
        if (map.has(source) && map.get(source) !== target) { valid = false; break; }
        map.set(source, target);
      }
    }
    if (!valid) continue;
    const test = transform(inputs.at(-1));
    if (test.some(line => line.some(value => !map.has(value)))) continue;
    predictions.push({ name, grid: test.map(line => line.map(value => map.get(value))) });
  }
  const distinct = new Map(predictions.map(entry => [canonical(entry.grid), entry.grid]));
  if (distinct.size !== 1) return { supported: false, reason: `D4+global-color-map rule is not unique (${distinct.size} output predictions)` };
  const inferred = [...distinct.values()][0];
  if (canonical(inferred) !== canonical(gold)) return { supported: false, reason: 'unique D4+global-color-map prediction does not match source target' };
  return { supported: true, answer: canonical(inferred), inferred_rules: predictions.map(entry => entry.name), demonstrations: outputs.length };
}

const gymRecords = selectedIds.reasoning_gym.map(id => sourceRecords.get(id));
const synlogicAudit = selectedIds.synlogic_candidates.map(id => {
  const row = sourceRecords.get(id), checked = checkSynlogic(row);
  const expectedSupported = synlogicSupported.has(id);
  if (checked.supported !== expectedSupported) throw new Error(`${id}: supported candidate membership changed (${checked.reason ?? 'unexpected support'})`);
  return { id, source_group: row.split_groups[0], split: row.split, license: row.license.spdx,
    disposition: checked.supported ? 'included' : 'omitted-unsupported-rule', checker: checked };
});
const gymSolved = gymRecords.map(row => {
  const question = row.consumer.context.find(message => message.role === 'user')?.content;
  if (!question || row.consumer.withheld?.includes('sources') !== true) throw new Error(`${row.id}: unexpected visible-source boundary`);
  const answer = solveRewrite(question), target = targetFinalSequence(row.target.value);
  if (answer !== target) throw new Error(`${row.id}: independently solved state differs from original label`);
  return { row, question, answer };
});

const taskCases = [];
function buildBatch(family, mode, entries) {
  const label = family === 'reasoning_gym' ? 'token-rewrite' : 'arc-grid';
  const caseId = `${family}-${mode}-${sha(entries.map(entry => entry.row.id).join('\n')).slice(0, 16)}`;
  const items = entries.map((entry, index) => ({ key: `item-${String(index + 1).padStart(2, '0')}`,
    source_id: entry.row.id, question: 'Solve the task described in the complete evidence.', original_question: entry.question,
    evidence: entry.question, answer: entry.answer,
    source: entry.row, answer_format: family === 'reasoning_gym' ?
      'Return only the terminal token sequence separated by single spaces. Return exactly empty if no tokens remain.' :
      'Return only the output grid as a compact JSON array of integer arrays. Do not include tags, reasoning, or a code fence.' }));
  const expected = Object.fromEntries(items.map(item => [item.key, item.answer]));
  const files = {
    'task.json': canonical({ instruction: `Read every item under items/ and solve the task in its question. For each item, return only the requested result in item.answer_format. Store one string answer per item filename stem in output_file, as specified by output_contract.`,
      answer_mode: mode, output_file: 'answers.json', items: items.map(item => `items/${item.key}.json`),
      output_contract: 'Write one JSON object with exactly one listed item filename stem as each key and the requested result string as its value.' }) + '\n',
    ...Object.fromEntries(items.map(item => [`items/${item.key}.json`, canonical({ source_record_id: item.source_id,
      question: item.question, evidence: item.evidence, answer_format: item.answer_format }) + '\n'])),
    'answers.json': '{}\n',
  };
  const expectedFiles = { ...files, 'answers.json': canonical(expected) + '\n' };
  const setupCode = `type Item = { source_record_id: string; question: string; evidence: string; answer_format: string };\n` +
    `type State = { cursor: number; answers: Record<string, string> };\n` +
    `const files = await folder.files('items/*.json');\n` +
    `const solveItem = nl<${mode === 'soft' ? 'Neuralese<string>' : 'string'}>\`Use only item.question and the complete item.evidence. Follow item.answer_format exactly and return only the result.\`;\n` +
    `const task = JSON.parse(await folder.file('task.json').readText()) as { output_file: string };\n` +
    `files.length;`;
  const code =
    `const step = async (state: State): Promise<State> => {\n` +
    `  if (state.cursor >= files.length) return state;\n` +
    `  const file = files[state.cursor];\n` +
    `  const item = JSON.parse(await file.readText()) as Item;\n` +
    `  const answer = await solveItem(item);\n` +
    (mode === 'soft' ? `  const answerText = String(answer);\n` : `  const answerText = answer;\n`) +
    `  const key = file.name.slice(0, -5);\n` +
    `  return { cursor: state.cursor + 1, answers: { ...state.answers, [key]: answerText } };\n` +
    `};\n` +
    `const final = await step.iterateOn({ cursor: 0, answers: {} } as State).withLimit({ maxSteps: files.length }).until(state => state.cursor >= files.length);\n` +
    `await folder.file(task.output_file).writeText(JSON.stringify(final.answers) + '\\n');\n` +
    `const saved = JSON.parse(await folder.file(task.output_file).readText()) as Record<string, string>;\n` +
    `if (JSON.stringify(saved) !== JSON.stringify(final.answers)) throw new Error('saved answers differ from final state');\n` +
    `return saved;`;
  const groupNames = [...new Set(items.flatMap(item => item.source.split_groups ?? []))].sort();
  const split = entries[0].row.split;
  if (entries.some(entry => entry.row.split !== split || entry.row.license.spdx !== entries[0].row.license.spdx))
    throw new Error(`${caseId}: mixed split or license batch`);
  const sourceBundleGroup = `s1-derived-batch:${sha(items.map(item => item.source_id).join('\n')).slice(0, 24)}`;
  const childCalls = items.map((item, index) => {
    const value = mode === 'soft' ? `<|neuralese|>${item.answer}<|/neuralese|>` : item.answer;
    const child = { match: `source_record_id: ${JSON.stringify(item.source_id)}`, source_binding: {
      path: `items/${item.key}.json`, sha256: sha(files[`items/${item.key}.json`]), source_record_id: item.source_id },
      calls: [returnCall(value)] };
    if (mode === 'soft') child.soft_output = { kind: 'Neuralese<string>', text: item.answer, readout_by_parent: 'String(answer)' };
    return child;
  });
  const sourceRows = items.map(item => ({ id: item.source_id, group: item.source.split_groups,
    split: item.source.split, license: item.source.license, label_origin: item.source.outcome?.label,
    upstream: item.source.lineage?.upstream, upstream_id: item.source.lineage?.upstream_id,
    prompt_sha256: sha(item.original_question), original_target_sha256: sha(item.source.target.value),
    independent_solution: item.answer }));
  const record = curriculumCase({
    family: `verified_reasoning_${family}`, familyVersion: 1, shape: caseId, variant: `${label}-iterate/1-${mode}`,
    splitGroup: sourceBundleGroup, split, slice: 'iterate', domain: family, mode: 'single_call', inline: 'required',
    iterate: 'required', evidence: { retrieved: items.map(item => item.source_id),
      background: [`S1 snapshot source family ${family}; original source groups are retained in dataset_records.`] },
    minimumSequence: ['read each task file at its iteration step', 'ask one inline child to solve that task',
      ...(mode === 'soft' ? ['read the typed Neuralese<string> result into a normal string'] : []),
      'carry answers in iterateOn state', 'write and read back answers.json'],
    reference: { root: [evalCall(setupCode), evalCall(code), returnCall(expected)], children: childCalls },
    root: { name: 'collect_answers', args: {}, returns: 'Record<string, string>', kind: 'directory-reducer',
      instructions: `Read task.json and each listed item file. Each item includes its question, complete evidence and answer_format; pass the full item to one inline child and solve only from those fields. ${mode === 'soft' ? 'Each child returns Neuralese<string>; convert it with String(answer) after the direct typed child call.' : 'Each child returns a crisp string.'} Carry each completed answer forward with iterateOn. After all items are solved, write the exact answer map to task.output_file and read it back. Return the saved map.` },
    files: { 'types.ts': `export type Item = { source_record_id: string; question: string; answer_format: string };\n` },
    inputs: {}, expected, expectedFiles, operation: 'verified-s1-reasoning-collection',
    folderFiles: Object.fromEntries(Object.entries(files).map(([path, body]) => [path, body])),
    worldSemantics: { family, answer_mode: mode, independent_world_credit: 0,
      source_groups: groupNames, solver: family === 'reasoning_gym' ? 'leftmost-token-rewrite/1' :
        'unique-output-within-d4-transform-plus-demonstration-derived-global-color-function/1' },
    decisive: ['The result follows from each visible question and its declared deterministic rules.'],
    plausibleActions: ['read source item', 'solve it using the question rules', 'record only the requested result'],
  });
  record.dataset = family;
  record.dataset_records = sourceRows;
  record.source_revisions = [`s1-full-final-20261003/${family}`];
  record.generation = { ...(record.generation ?? {}), ...record.generation,
    generator: 'natlang.verified_reasoning_collection/1', source_snapshot: { id: 's1-full-final-20261003',
      manifest_sha256: sha(corpusManifestBytes), family_file: familyFiles[family], file_sha256: familyDigests[family].sha256 },
    original_source_groups: groupNames, task_variant: true, independent_world_credit: 0,
    answer_payload_mode: mode, target_values_visible_to_model: false, provider_calls: 0,
    teacher_observations: 0, training_admission: false };
  record.semantics.oracle = 'exact';
  record.semantics.files_oracle = { compare: 'json-string-record' };
  taskCases.push(record);
}

for (let i = 0; i < gymSolved.length; i += 4) buildBatch('reasoning_gym', i < 8 ? 'crisp' : 'soft', gymSolved.slice(i, i + 4));
const synlogicSolved = synlogicAudit.filter(entry => entry.disposition === 'included').map(entry => {
  const row = sourceRecords.get(entry.id);
  const question = row.consumer.context.find(message => message.role === 'user').content;
  return { row, question, answer: entry.checker.answer };
});
for (let i = 0; i < synlogicSolved.length; i += 4) buildBatch('reasoning_synlogic', i === 0 ? 'crisp' : 'soft', synlogicSolved.slice(i, i + 4));

const sourceText = taskCases.map(record => JSON.stringify(record)).join('\n') + '\n';
const sourceSha = sha(sourceText);
const sourceProof = { schema: 's1-verified-reasoning-pools-proof/1', source_cases_sha256: sourceSha,
  corpus_manifest_sha256: sha(corpusManifestBytes), family_files: familyDigests,
  selected_source_records: gymSolved.length + synlogicAudit.filter(entry => entry.disposition === 'included').length,
  source_task_variants: taskCases.length, reasoning_gym: { source_records: gymSolved.map(entry => ({ id: entry.row.id,
    group: entry.row.split_groups, split: entry.row.split, license: entry.row.license.spdx,
    source_question_sha256: sha(entry.question), target_sha256: sha(entry.row.target.value),
    independently_computed_terminal_state: entry.answer })), checker: 'leftmost adjacent directed rewrite until no rule applies; source rules parsed from prompt',
    independently_solved: gymSolved.length, target_matches: gymSolved.length },
  synlogic: { candidate_pool: synlogicAudit, included: synlogicAudit.filter(entry => entry.disposition === 'included').length,
    omitted_unsupported: synlogicAudit.filter(entry => entry.disposition !== 'included').length,
    checker: 'all D4 rotations/reflections plus a global integer color function inferred from every visible demonstration; requires full test-color coverage, one unique predicted output, and exact equality to target grid' },
  split_counts: { train: taskCases.filter(record => record.split === 'train').length },
  task_modes: Object.fromEntries(['crisp', 'soft'].map(mode => [mode, taskCases.filter(record => record.generation.answer_payload_mode === mode).length])),
  no_hidden_targets_in_prompts: true, source_group_preservation: true, provider_calls: 0, teacher_observations: 0,
  independent_world_credit: 0, training_admission: false, generation_admission: 'pending root review' };
await mkdir(out, { recursive: false });
const selection = { schema: 'natlang.verified_reasoning_selection/1', corpus: 's1-full-final-20261003',
  source_ids: { reasoning_gym: selectedIds.reasoning_gym,
    synlogic_candidates: selectedIds.synlogic_candidates,
    synlogic_included: synlogicAudit.filter(entry => entry.disposition === 'included').map(entry => entry.id),
    synlogic_omitted: synlogicAudit.filter(entry => entry.disposition !== 'included').map(entry => entry.id) },
  reasoning_gym_answer_modes: { crisp: selectedIds.reasoning_gym.slice(0, 8), soft: selectedIds.reasoning_gym.slice(8) },
  synlogic_answer_modes: { crisp: synlogicSolved.slice(0, 4).map(entry => entry.row.id), soft: synlogicSolved.slice(4).map(entry => entry.row.id) },
  no_new_independent_world_credit: true };
await writeFile(resolve(out, 'selection.json'), JSON.stringify(selection, null, 2) + '\n', { flag: 'wx' });
await writeFile(resolve(out, 'source.cases.jsonl'), sourceText, { flag: 'wx' });
await writeFile(resolve(out, 'source-proof.json'), JSON.stringify(sourceProof, null, 2) + '\n', { flag: 'wx' });
await writeFile(resolve(out, 'source-manifest.json'), JSON.stringify({ schema: 'natlang.verified_reasoning_pools/1',
  source_cases: 'source.cases.jsonl', source_cases_sha256: sourceSha, selection: 'selection.json',
  selection_sha256: sha(JSON.stringify(selection, null, 2) + '\n'), source_proof: 'source-proof.json',
  source_proof_sha256: sha(JSON.stringify(sourceProof, null, 2) + '\n'), corpus: 'data/neuralese/corpora/s1-full-final-20261003',
  corpus_manifest_sha256: sha(corpusManifestBytes), builder: 'ts-host/scripts/inline-curriculum/build-verified-reasoning-pools-v1.mjs',
  builder_sha256: sha(await readFile(new URL('./build-verified-reasoning-pools-v1.mjs', import.meta.url))),
  generation_admission: 'pending root review', training_admission: false }, null, 2) + '\n', { flag: 'wx' });
console.log(JSON.stringify({ out, source_sha256: sourceSha, source_task_variants: taskCases.length,
  reasoning_gym_source_records: gymSolved.length, synlogic_candidates: synlogicAudit.length,
  synlogic_included: synlogicSolved.length, synlogic_omitted: synlogicAudit.length - synlogicSolved.length,
  family_files: familyDigests, provider_calls: 0 }, null, 2));
