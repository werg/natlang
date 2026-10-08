#!/usr/bin/env node
// Build small, source-derived task pools whose labels are checked by independent host solvers.
// The output is a task variant over existing S1 records; it creates no new source-world credit.
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { curriculumCase, evalCall, returnCall } from './lib.mjs';

const { values } = parseArgs({ options: { corpus: { type: 'string' }, out: { type: 'string' }, selection: { type: 'string' } } });
if (!values.corpus || !values.out) throw new Error('usage: build-verified-reasoning-pools-v1.mjs --corpus DIR --out NEW_DIR [--selection selection.json]');
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

const defaultSelectedIds = {
  reasoning_gym: [0, 1, 2, 3, 4, 5, 6, 8, 9, 11, 14, 18, 19, 22, 23, 24]
    .map(index => `sdkb:reasoning-gym:reasoning_gym-${index}`),
  synlogic_candidates: [],
};
defaultSelectedIds.reasoning_gym_answer_modes = {
  crisp: defaultSelectedIds.reasoning_gym.slice(0, 8),
  soft: defaultSelectedIds.reasoning_gym.slice(8),
  soft_carry: [],
};
const synlogicSupported = new Set([
  'sdkb:synlogic:synlogic-easy-train-1', 'sdkb:synlogic:synlogic-easy-train-28',
  'sdkb:synlogic:synlogic-easy-train-153', 'sdkb:synlogic:synlogic-easy-train-207',
  'sdkb:synlogic:synlogic-easy-train-263', 'sdkb:synlogic:synlogic-easy-train-288',
  'sdkb:synlogic:synlogic-easy-train-308',
]);
// Candidate pool includes the seven demonstration-checkable grids and nine explicit omissions.
// The unsupported records are documented, not silently added to the generated cases.
defaultSelectedIds.synlogic_candidates = [
  'sdkb:synlogic:synlogic-easy-train-0', 'sdkb:synlogic:synlogic-easy-train-1',
  'sdkb:synlogic:synlogic-easy-train-2', 'sdkb:synlogic:synlogic-easy-train-3',
  'sdkb:synlogic:synlogic-easy-train-4', 'sdkb:synlogic:synlogic-easy-train-5',
  'sdkb:synlogic:synlogic-easy-train-6', 'sdkb:synlogic:synlogic-easy-train-8',
  'sdkb:synlogic:synlogic-easy-train-9', 'sdkb:synlogic:synlogic-easy-train-11',
  'sdkb:synlogic:synlogic-easy-train-28', 'sdkb:synlogic:synlogic-easy-train-153',
  'sdkb:synlogic:synlogic-easy-train-207', 'sdkb:synlogic:synlogic-easy-train-263',
  'sdkb:synlogic:synlogic-easy-train-288', 'sdkb:synlogic:synlogic-easy-train-308',
];
const selectionPath = values.selection ? resolve(values.selection) : null;
const selectionInput = selectionPath ? JSON.parse(await readFile(selectionPath, 'utf8')) : null;
if (selectionInput && (selectionInput.schema !== 'natlang.verified_reasoning_selection/1' ||
    selectionInput.corpus !== 's1-full-final-20261003'))
  throw new Error('selection schema or pinned source corpus is unsupported');
const selectedIds = selectionInput ? {
  reasoning_gym: selectionInput.source_ids?.reasoning_gym,
  reasoning_gym_by_solver: selectionInput.reasoning_gym_by_solver ?? {},
  synlogic_candidates: selectionInput.source_ids?.synlogic_candidates ?? [],
  reasoning_gym_answer_modes: selectionInput.reasoning_gym_answer_modes,
} : { ...defaultSelectedIds, reasoning_gym_by_solver: {} };
if (!Array.isArray(selectedIds.reasoning_gym) ||
    !Array.isArray(selectedIds.synlogic_candidates) || !selectedIds.reasoning_gym_answer_modes ||
    !Array.isArray(selectedIds.reasoning_gym_answer_modes.crisp) ||
    !Array.isArray(selectedIds.reasoning_gym_answer_modes.soft))
  throw new Error('selection must declare source_ids.reasoning_gym, source_ids.synlogic_candidates, and crisp/soft reasoning_gym_answer_modes');
const gymModeIds = [...selectedIds.reasoning_gym_answer_modes.crisp, ...selectedIds.reasoning_gym_answer_modes.soft];
gymModeIds.push(...(selectedIds.reasoning_gym_answer_modes.soft_carry ?? []));
if (new Set(selectedIds.reasoning_gym).size !== selectedIds.reasoning_gym.length ||
    new Set(gymModeIds).size !== gymModeIds.length ||
    canonical([...selectedIds.reasoning_gym].sort()) !== canonical([...gymModeIds].sort()) ||
    selectedIds.reasoning_gym_answer_modes.crisp.length % 4 !== 0 ||
    selectedIds.reasoning_gym_answer_modes.soft.length % 4 !== 0 ||
    (selectedIds.reasoning_gym_answer_modes.soft_carry ?? []).length % 4 !== 0)
  throw new Error('selection must assign each unique reasoning-gym source exactly once and provide complete four-item crisp/soft batches');
const solverRows = Object.entries(selectedIds.reasoning_gym_by_solver).flatMap(([solver, modes]) =>
  Object.entries(modes).flatMap(([mode, ids]) => ids.map(id => ({ id, solver, mode }))));
const solverIds = solverRows.map(row => row.id);
if (new Set(solverIds).size !== solverIds.length || solverRows.some(row =>
    !['base_conversion', 'gcd', 'fibonacci', 'count_bits', 'spell_backward', 'word_sequence_reversal', 'word_sorting', 'modular_inverse', 'basic_arithmetic'].includes(row.solver) ||
    !['crisp', 'soft', 'soft_carry'].includes(row.mode) || !Array.isArray(selectedIds.reasoning_gym_by_solver[row.solver][row.mode])))
  throw new Error('additional solver selections must use unique IDs, supported solvers, and explicit crisp/soft/soft_carry modes');
if (solverRows.some(row => selectedIds.reasoning_gym.includes(row.id) || !row.id.startsWith('sdkb:reasoning-gym:')))
  throw new Error('additional solver IDs must be reasoning_gym rows disjoint from rewrite IDs');
for (const [solver, modes] of Object.entries(selectedIds.reasoning_gym_by_solver)) {
  if (!modes || typeof modes !== 'object' || Array.isArray(modes)) throw new Error(`${solver}: solver modes must be an object`);
  for (const mode of ['crisp', 'soft', 'soft_carry']) {
    const ids = modes[mode] ?? [];
    if (!Array.isArray(ids) || ids.length % 4 !== 0) throw new Error(`${solver}.${mode} must contain zero or a multiple of four IDs`);
  }
  if (Object.keys(modes).some(mode => !['crisp', 'soft', 'soft_carry'].includes(mode)))
    throw new Error(`${solver}: unknown answer mode`);
}
const familyFiles = {
  reasoning_gym: 'reasoning_gym.port-records.jsonl',
  reasoning_synlogic: 'reasoning_synlogic.port-records.jsonl',
};
const wanted = new Set([...selectedIds.reasoning_gym, ...solverIds, ...selectedIds.synlogic_candidates]);
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
  for (;;) {
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
}
function sourceTaskFamily(row) {
  for (const source of row.sources ?? []) {
    const match = source.text?.match(/Worked example \(([^)]+)\):/);
    if (match) return match[1];
  }
  return null;
}
function solveGymFamily(family, question) {
  if (family === 'base_conversion') {
    const match = question.match(/base-(\d+)\s+number\s+([0-9a-z]+)\s+to\s+base-(\d+)/i);
    if (!match) throw new Error('base-conversion prompt shape is unsupported');
    const [, sourceBaseText, digitsText, targetBaseText] = match;
    const sourceBase = Number(sourceBaseText), targetBase = Number(targetBaseText);
    if (sourceBase < 2 || sourceBase > 36 || targetBase < 2 || targetBase > 36)
      throw new Error('base-conversion bases are outside the supported range 2..36');
    let value = 0n;
    for (const char of digitsText.toLowerCase()) {
      const digit = '0123456789abcdefghijklmnopqrstuvwxyz'.indexOf(char);
      if (digit < 0 || digit >= sourceBase) throw new Error('digit is invalid for source base');
      value = value * BigInt(sourceBase) + BigInt(digit);
    }
    if (value === 0n) return '0';
    let result = '';
    const base = BigInt(targetBase), chars = '0123456789abcdefghijklmnopqrstuvwxyz';
    while (value > 0n) { result = chars[Number(value % base)] + result; value /= base; }
    return result;
  }
  if (family === 'gcd') {
    const numbers = [...question.matchAll(/(?<![\w.])-?\d+/g)].map(match => BigInt(match[0]));
    if (numbers.length !== 2) throw new Error('GCD prompt must contain exactly two integers');
    let [a, b] = numbers.map(value => value < 0n ? -value : value);
    while (b !== 0n) [a, b] = [b, a % b];
    return String(a);
  }
  if (family === 'fibonacci') {
    const matches = [...question.matchAll(/(\d+)-?(?:st|nd|rd|th)\s+Fibonacci number/gi)];
    if (matches.length !== 1 || !/F\(0\)\s*=\s*0/i.test(question) || !/F\(1\)\s*=\s*1/i.test(question))
      throw new Error('Fibonacci prompt must state one index and both base cases');
    const n = Number(matches[0][1]);
    if (!Number.isSafeInteger(n) || n < 0 || n > 5000) throw new Error('Fibonacci index is outside supported exact range');
    let a = 0n, b = 1n;
    for (let index = 0; index < n; index++) [a, b] = [b, a + b];
    return String(a);
  }
  if (family === 'count_bits') {
    const match = question.match(/binary representation of the number\s+(\d+)/i);
    if (!match) throw new Error('bit-count prompt must state one nonnegative decimal integer');
    let value = BigInt(match[1]), count = 0;
    while (value) { count += Number(value & 1n); value >>= 1n; }
    return String(count);
  }
  if (family === 'spell_backward') {
    const matches = [...question.matchAll(/Spell this word backward \(example: [^\n]+\):\s*([A-Za-z]+)/gi)];
    if (matches.length !== 1) throw new Error('backward-spelling prompt shape is unsupported');
    return [...matches[0][1]].reverse().join('');
  }
  if (family === 'word_sequence_reversal') {
    const matches = [...question.matchAll(/Reverse this list of words:\s*([^\n]+)/gi)];
    if (matches.length !== 1 || !/comma-separated list of words/i.test(question))
      throw new Error('word-sequence reversal prompt shape is unsupported');
    const words = matches[0][1].split(/,\s*/).map(word => word.trim());
    if (words.length < 2 || words.some(word => !/^[\p{L}\p{N}_'-]+$/u.test(word)))
      throw new Error('word-sequence list contains unsupported tokens');
    return words.reverse().join(', ');
  }
  if (family === 'word_sorting') {
    const matches = [...question.matchAll(/Words:\s*\n?([^\n]+)/gi)];
    const direction = question.match(/\b(ascending|descending) order\b/i)?.[1]?.toLowerCase();
    if (matches.length !== 1 || !/ASCII\/Unicode ordering/i.test(question) || !direction)
      throw new Error('word-sorting prompt shape is unsupported');
    const words = matches[0][1].split(/,\s*/).map(word => word.trim());
    if (words.length < 2 || words.some(word => !/^[\p{L}\p{N}_'-]+$/u.test(word)))
      throw new Error('word-sorting list contains unsupported tokens');
    return words.sort((a, b) => (a < b ? -1 : a > b ? 1 : 0) * (direction === 'ascending' ? 1 : -1)).join(', ');
  }
  if (family === 'modular_inverse') {
    const match = question.match(/integers\s+a\s*=\s*(-?\d+)\s+and\s+modulus\s+m\s*=\s*(\d+).*?smallest nonnegative/i);
    if (!match) throw new Error('modular-inverse prompt shape is unsupported');
    const a = Number(match[1]), m = Number(match[2]);
    if (!Number.isSafeInteger(a) || !Number.isSafeInteger(m) || m < 2 || gcdNumber(a, m) !== 1)
      throw new Error('modular inverse is not uniquely defined for these inputs');
    for (let x = 0; x < m; x++) if ((a * x) % m === 1 % m) return String(x);
    throw new Error('modular inverse not found');
  }
  throw new Error(`unsupported Reasoning Gym solver family: ${family}`);
}
function gcdNumber(a, b) {
  a = Math.abs(a); b = Math.abs(b);
  while (b !== 0) [a, b] = [b, a % b];
  return a;
}
function finalGymAnswer(target) {
  const text = target.split('</think>').at(-1).trim();
  const match = text.match(/(?:final answer(?: is)?|correct answer(?: is)?|answer(?: is)?|modular inverse is)\s*:?\s*(.+?)\s*\.?$/i);
  if (!match) throw new Error('source target has no supported final answer form');
  return match[1].trim().replace(/^\$|\$$/g, '').replace(/\.$/, '').trim();
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
const modeByGymId = new Map(selectedIds.reasoning_gym_answer_modes.crisp.map(id => [id, 'crisp']));
for (const id of selectedIds.reasoning_gym_answer_modes.soft) modeByGymId.set(id, 'soft');
for (const id of selectedIds.reasoning_gym_answer_modes.soft_carry ?? []) modeByGymId.set(id, 'soft_carry');
const gymSolved = gymRecords.map(row => {
  const question = row.consumer.context.find(message => message.role === 'user')?.content;
  if (!question || row.consumer.withheld?.includes('sources') !== true) throw new Error(`${row.id}: unexpected visible-source boundary`);
  const answer = solveRewrite(question), target = targetFinalSequence(row.target.value);
  if (answer !== target) throw new Error(`${row.id}: independently solved state differs from original label`);
  return { row, question, answer, mode: modeByGymId.get(row.id), solver: 'token_rewrite' };
});
const additionalGymSolved = solverRows.map(({ id, solver, mode }) => {
  const row = sourceRecords.get(id), question = row.consumer.context.find(message => message.role === 'user')?.content;
  if (sourceTaskFamily(row) !== solver) throw new Error(`${id}: source worked-example family is not ${solver}`);
  if (!question || row.consumer.withheld?.includes('sources') !== true) throw new Error(`${id}: unexpected visible-source boundary`);
  const answer = solveGymFamily(solver, question), target = finalGymAnswer(row.target.value);
  if (answer !== target) throw new Error(`${id}: independent ${solver} solver differs from original label (${answer} vs ${target})`);
  return { row, question, answer, mode, solver };
});

const taskCases = [];
function buildBatch(family, mode, entries, solver = null) {
  const label = solver === 'token_rewrite' ? 'token-rewrite' : solver ?? 'arc-grid';
  const caseId = solver && solver !== 'token_rewrite' ?
    `${family}-${label}-${mode}-${sha(entries.map(entry => entry.row.id).join('\n')).slice(0, 16)}` :
    `${family}-${mode}-${sha(entries.map(entry => entry.row.id).join('\n')).slice(0, 16)}`;
  const items = entries.map((entry, index) => ({ key: `item-${String(index + 1).padStart(2, '0')}`,
    source_id: entry.row.id, question: 'Solve the task described in the complete evidence.', original_question: entry.question,
    evidence: entry.question, answer: entry.answer,
    source: entry.row, answer_format: family === 'reasoning_gym' ? (solver === 'token_rewrite' ?
      'Return only the terminal token sequence separated by single spaces. Return exactly empty if no tokens remain.' :
      `Return only the ${solver.replaceAll('_', ' ')} result requested by the question. Preserve its exact answer type, case, and delimiter.`) :
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
  const carryMode = mode === 'soft_carry';
  const setupCode = `type Item = { source_record_id: string; question: string; evidence: string; answer_format: string };\n` +
    (carryMode ? `type State = { cursor: number; notes: Neuralese<string> };\n` : `type State = { cursor: number; answers: Record<string, string> };\n`) +
    `const files = await folder.files('items/*.json');\n` +
    (carryMode ?
      `const seedNotes: Neuralese<() => Promise<Neuralese<string>>> = nl.with<Neuralese<string>>({})\`Create the initial carried answer map. Return exactly the empty JSON object {} as text.\`;\n` +
      `const updateItem: Neuralese<(input: { item: Item; priorAnswers: Neuralese<string> }) => Promise<Neuralese<string>>> = nl.with<Neuralese<string>>({})\`You receive item and priorAnswers. Read priorAnswers as String(input.priorAnswers), parse that JSON object, and keep every existing key and value unchanged. Use only item.question and the complete item.evidence to solve this item. Add exactly this item's filename stem as a new key with its requested answer string. Return the complete updated JSON object as text.\`;\n` :
      `const solveItem = nl<${mode === 'crisp' ? 'string' : 'Neuralese<string>'}>\`Use only item.question and the complete item.evidence. Follow item.answer_format exactly and return only the result.\`;\n`) +
    `const task = JSON.parse(await folder.file('task.json').readText()) as { output_file: string };\n` +
    `files.length;`;
  const code = carryMode ?
    `const seed = await seedNotes();\n` +
    `const step = async (state: State): Promise<State> => {\n` +
    `  if (state.cursor >= files.length) return state;\n` +
    `  const file = files[state.cursor];\n` +
    `  const item = JSON.parse(await file.readText()) as Item;\n` +
    `  const notes = await updateItem({ item, priorAnswers: state.notes });\n` +
    `  return { cursor: state.cursor + 1, notes };\n` +
    `};\n` +
    `const final = await step.iterateOn({ cursor: 0, notes: seed } as State).withLimit({ maxSteps: files.length }).until(state => state.cursor >= files.length);\n` +
    `const completed = JSON.parse(String(final.notes)) as Record<string, string>;\n` +
    `await folder.file(task.output_file).writeText(JSON.stringify(completed) + '\\n');\n` +
    `const saved = JSON.parse(await folder.file(task.output_file).readText()) as Record<string, string>;\n` +
    `if (JSON.stringify(saved) !== JSON.stringify(completed)) throw new Error('saved answers differ from final carried notes');\n` +
    `return saved;` :
    `const step = async (state: State): Promise<State> => {\n` +
    `  if (state.cursor >= files.length) return state;\n` +
    `  const file = files[state.cursor];\n` +
    `  const item = JSON.parse(await file.readText()) as Item;\n` +
    `  const answer = await solveItem(item);\n` +
    (mode !== 'crisp' ? `  const answerText = String(answer);\n` : `  const answerText = answer;\n`) +
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
  const carryByKey = {};
  const itemChildCalls = items.map((item, index) => {
    if (carryMode) carryByKey[item.key] = item.answer;
    const softText = carryMode ? canonical(carryByKey) : item.answer;
    const value = mode !== 'crisp' ? `<|neuralese|>${softText}<|/neuralese|>` : item.answer;
    const child = { match: `source_record_id: ${JSON.stringify(item.source_id)}`, source_binding: {
      path: `items/${item.key}.json`, sha256: sha(files[`items/${item.key}.json`]), source_record_id: item.source_id },
      calls: [returnCall(value)] };
    if (mode !== 'crisp') child.soft_output = { kind: 'Neuralese<string>', text: softText,
      ...(carryMode && index === items.length - 1 ? { readout_by_parent: 'String(final.notes)', carried_keys: Object.keys(carryByKey) } :
        carryMode ? { argument_mode: 'typed_argument', next_argument: 'input.priorAnswers', carried_keys: Object.keys(carryByKey) } :
        { readout_by_parent: 'String(answer)' }) };
    if (carryMode) child.expected_soft_input = 'input.priorAnswers';
    return child;
  });
  const childCalls = carryMode ? [{ match: 'initial carried answer map', calls: [returnCall('<|neuralese|>{}<|/neuralese|>')],
    soft_output: { kind: 'Neuralese<string>', text: '{}', argument_mode: 'typed_argument',
      next_argument: 'input.priorAnswers', carried_keys: [] } },
    ...itemChildCalls] : itemChildCalls;
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
      ...(carryMode ? ['pass the previous typed Neuralese<string> answer map to the next child', 'read the complete final typed answer map once'] :
        mode !== 'crisp' ? ['read the typed Neuralese<string> result into a normal string'] : []),
      'carry answers in iterateOn state', 'write and read back answers.json'],
    reference: { root: [evalCall(setupCode), evalCall(code), returnCall(expected)], children: childCalls },
    root: { name: 'collect_answers', args: {}, returns: 'Record<string, string>', kind: 'directory-reducer',
      instructions: `Read task.json and each listed item file. Each item includes its question, complete evidence and answer_format; pass the full item to one inline child and solve only from those fields. ${carryMode ? 'For each item after the first, pass the prior typed Neuralese<string> answer map as an argument to the child. The child must preserve all earlier entries and add only the current item. Keep that typed map in iterateOn state. Read it as a string only after the last item, then write and read back the saved map.' : mode !== 'crisp' ? 'Each child returns Neuralese<string>; convert it with String(answer) after the direct typed child call.' : 'Each child returns a crisp string.'} ${carryMode ? '' : 'Carry each completed answer forward with iterateOn. After all items are solved, write the exact answer map to task.output_file and read it back. Return the saved map.'}` },
    files: { 'types.ts': `export type Item = { source_record_id: string; question: string; answer_format: string };\n` },
    inputs: {}, expected, expectedFiles, operation: 'verified-s1-reasoning-collection',
    folderFiles: Object.fromEntries(Object.entries(files).map(([path, body]) => [path, body])),
    worldSemantics: { family, answer_mode: mode, independent_world_credit: 0,
      source_groups: groupNames, solver: solver === 'token_rewrite' ? 'leftmost-token-rewrite/1' : solver ??
        'unique-output-within-d4-transform-plus-demonstration-derived-global-color-function/1' },
    decisive: ['The result follows from each visible question and its declared deterministic rules.'],
    plausibleActions: ['read source item', 'solve it using the question rules', 'record only the requested result'],
  });
  record.dataset = family;
  record.source_groups = groupNames;
  record.dataset_records = sourceRows;
  record.source_revisions = [`s1-full-final-20261003/${family}`];
  record.generation = { ...(record.generation ?? {}), ...record.generation,
    generator: 'natlang.verified_reasoning_collection/1', source_snapshot: { id: 's1-full-final-20261003',
      manifest_sha256: sha(corpusManifestBytes), family_file: familyFiles[family], file_sha256: familyDigests[family].sha256 },
    original_source_groups: groupNames, source_bundle_group: sourceBundleGroup,
    task_variant: true, independent_world_credit: 0,
    answer_payload_mode: mode, target_values_visible_to_model: false, provider_calls: 0,
    teacher_observations: 0, training_admission: false };
  record.semantics.oracle = 'exact';
  record.semantics.files_oracle = { compare: 'json-string-record' };
  taskCases.push(record);
}

const allGymSolved = [...gymSolved, ...additionalGymSolved];
const solverNames = [...new Set(allGymSolved.map(entry => entry.solver))];
for (const solver of solverNames) for (const mode of ['crisp', 'soft', 'soft_carry']) {
  const entries = allGymSolved.filter(entry => entry.solver === solver && entry.mode === mode);
  for (let i = 0; i < entries.length; i += 4) buildBatch('reasoning_gym', mode, entries.slice(i, i + 4), solver);
}
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
  selected_source_records: allGymSolved.length + synlogicAudit.filter(entry => entry.disposition === 'included').length,
  source_task_variants: taskCases.length, reasoning_gym: { source_records: allGymSolved.map(entry => ({ id: entry.row.id, solver: entry.solver,
    group: entry.row.split_groups, split: entry.row.split, license: entry.row.license.spdx,
    source_question_sha256: sha(entry.question), target_sha256: sha(entry.row.target.value),
    independently_computed_terminal_state: entry.answer })), checker: 'each selected source is solved by the named family-specific prompt parser and deterministic solver; source label is checked separately',
    independently_solved: allGymSolved.length, target_matches: allGymSolved.length },
  synlogic: { candidate_pool: synlogicAudit, included: synlogicAudit.filter(entry => entry.disposition === 'included').length,
    omitted_unsupported: synlogicAudit.filter(entry => entry.disposition !== 'included').length,
    checker: 'all D4 rotations/reflections plus a global integer color function inferred from every visible demonstration; requires full test-color coverage, one unique predicted output, and exact equality to target grid' },
  split_counts: { train: taskCases.filter(record => record.split === 'train').length },
  task_modes: Object.fromEntries(['crisp', 'soft', 'soft_carry'].map(mode => [mode, taskCases.filter(record => record.generation.answer_payload_mode === mode).length])),
  no_hidden_targets_in_prompts: true, source_group_preservation: true, provider_calls: 0, teacher_observations: 0,
  independent_world_credit: 0, training_admission: false, generation_admission: 'pending root review' };
await mkdir(out, { recursive: false });
const selection = { schema: 'natlang.verified_reasoning_selection/1', corpus: 's1-full-final-20261003',
  source_ids: { reasoning_gym: selectedIds.reasoning_gym,
    synlogic_candidates: selectedIds.synlogic_candidates,
    synlogic_included: synlogicAudit.filter(entry => entry.disposition === 'included').map(entry => entry.id),
    synlogic_omitted: synlogicAudit.filter(entry => entry.disposition !== 'included').map(entry => entry.id) },
  reasoning_gym_answer_modes: selectedIds.reasoning_gym_answer_modes,
  reasoning_gym_by_solver: selectedIds.reasoning_gym_by_solver,
  synlogic_answer_modes: { crisp: synlogicSolved.slice(0, 4).map(entry => entry.row.id), soft: synlogicSolved.slice(4).map(entry => entry.row.id) },
  selection_input_sha256: selectionPath ? sha(await readFile(selectionPath)) : null,
  omissions: selectionInput?.omissions ?? { synlogic: synlogicAudit.filter(entry => entry.disposition !== 'included') },
  source_metrics: selectionInput?.source_metrics ?? null,
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
  reasoning_gym_source_records: allGymSolved.length, synlogic_candidates: synlogicAudit.length,
  synlogic_included: synlogicSolved.length, synlogic_omitted: synlogicAudit.length - synlogicSolved.length,
  family_files: familyDigests, provider_calls: 0 }, null, 2));
