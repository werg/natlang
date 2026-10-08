#!/usr/bin/env node
// Independent audit of a generated S1 question-collection pool against its pinned source records.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { parseArgs } from 'node:util';

const { values } = parseArgs({ options: { source: { type: 'string' }, corpus: { type: 'string' }, out: { type: 'string' } } });
if (!values.source || !values.corpus || !values.out) throw new Error('usage: verify-s1-question-collection-v1.mjs --source CASES --corpus DIR --out REPORT.json');
const sourcePath = resolve(values.source), corpus = resolve(values.corpus), outPath = resolve(values.out);
const sha = value => createHash('sha256').update(value).digest('hex');
async function* lines(stream, hash) {
  let pending = Buffer.alloc(0);
  for await (const chunk of stream) {
    hash.update(chunk);
    pending = pending.length ? Buffer.concat([pending, chunk]) : chunk;
    let index;
    while ((index = pending.indexOf(0x0a)) >= 0) {
      let line = pending.subarray(0, index);
      if (line.at(-1) === 0x0d) line = line.subarray(0, -1);
      yield line.toString('utf8');
      pending = pending.subarray(index + 1);
    }
  }
  if (pending.length) yield pending.toString('utf8');
}
const bytes = await readFile(sourcePath), cases = bytes.toString('utf8').trimEnd().split('\n').map(line => JSON.parse(line));
const sourceIds = [...new Set(cases.flatMap(row => row.source_ids))];
const sourceRows = new Map(), filePins = new Map();
for (const family of [...new Set(cases.map(row => row.dataset))]) {
  const path = resolve(corpus, `${family}.port-records.jsonl`), hash = createHash('sha256');
  for await (const line of lines(createReadStream(path), hash)) {
    if (!line) continue;
    const row = JSON.parse(line);
    if (sourceIds.includes(row.id)) sourceRows.set(row.id, row);
  }
  filePins.set(family, hash.digest('hex'));
}
assert.equal(sourceRows.size, sourceIds.length, 'every selected source record must exist exactly once in its source family');
const manifestBytes = await readFile(resolve(corpus, 'manifest.json'));
const seenGroups = new Set(), seenIds = new Set();
for (const row of cases) {
  const expected = row.semantics.expected;
  const fileExpected = JSON.parse(row.semantics.expected_files['answers.json']);
  assert.deepEqual(fileExpected, expected, `${row.id}: answers.json must match the exact source target map`);
  assert.deepEqual(row.source_ids, row.dataset_records, `${row.id}: source record IDs must be preserved exactly`);
  assert.equal(row.source_revisions[0], `s1-full-final-20261003:manifest:${sha(manifestBytes)}`);
  const currentSources = row.dataset_records.map(id => sourceRows.get(id));
  assert(currentSources.every(source => source && source.family === row.dataset && source.split === row.split), `${row.id}: source family/split mismatch`);
  const groups = [...new Set(currentSources.flatMap(source => source.split_groups ?? []))].sort();
  assert.deepEqual(row.source_groups, groups, `${row.id}: original source groups must be preserved`);
  assert.equal(row.external_source.port_record_file_sha256, filePins.get(row.dataset), `${row.id}: source file SHA mismatch`);
  assert.equal(row.external_source.corpus_manifest_sha256, sha(manifestBytes));
  const itemPaths = Object.keys(row.semantics.folder_files).filter(path => path.startsWith('items/')).sort();
  assert.equal(itemPaths.length, currentSources.length, `${row.id}: item/source count mismatch`);
  const refs = row.curriculum.reference.children;
  assert.equal(refs.length, itemPaths.length, `${row.id}: one source-bound reference child is required per item`);
  const rootCode = row.curriculum.reference.root.find(call => call[0] === 'eval')?.[1]?.code ?? '';
  assert(rootCode.includes('JSON.parse(await file.readText())'), `${row.id}: root must read each item file before passing its content`);
  const boundPaths = new Set();
  for (const ref of refs) {
    const binding = ref.source_binding;
    assert(binding && typeof binding.path === 'string' && typeof binding.sha256 === 'string' &&
      typeof binding.source_record_id === 'string', `${row.id}: child reference must bind by read path, content hash, and source ID`);
    const body = row.semantics.folder_files[binding.path];
    assert.equal(typeof body, 'string', `${row.id}: source-bound reference path must exist`);
    assert.equal(sha(body), binding.sha256, `${row.id}: source-bound reference hash differs from the actual item bytes`);
    assert.equal(JSON.parse(body).source_record_id, binding.source_record_id, `${row.id}: source-bound ID differs from item content`);
    assert(!boundPaths.has(binding.path), `${row.id}: duplicate source-bound item path`);
    boundPaths.add(binding.path);
    assert.equal(ref.match, `source_record_id: ${JSON.stringify(binding.source_record_id)}`,
      `${row.id}: scripted child match must use the exact ID carried by the item`);
  }
  assert.deepEqual([...boundPaths].sort(), itemPaths, `${row.id}: every item needs exactly one source-bound reference child`);
  const prompts = new Set();
  for (let index = 0; index < currentSources.length; index++) {
    const source = currentSources[index], path = itemPaths[index], item = JSON.parse(row.semantics.folder_files[path]);
    const question = source.consumer?.context?.filter(message => message.role === 'user').map(message => message.content).join('\n').trim();
    const evidence = source.sources.map(item => `${item.role}:\n${item.text}`).join('\n\n');
    const key = path.slice('items/'.length, -'.json'.length);
    assert.deepEqual(Object.keys(item).sort(), ['answer_format', 'evidence', 'question', 'source_record_id'].sort(), `${row.id}/${path}: no gold-bearing payload keys`);
    assert.equal(item.source_record_id, source.id);
    assert.equal(item.question, question, `${row.id}/${path}: source question changed`);
    assert.equal(item.evidence, evidence, `${row.id}/${path}: visible source evidence changed`);
    assert.equal(expected[key], source.target.value, `${row.id}/${path}: gold differs from the original source target`);
    const updatedCollection = ['/3', '/4', '/5', '/6'].some(version => row.generation.generator.endsWith(version));
    const qaMap = ['qa_extractive', 'qa_multihop'].includes(source.family) && updatedCollection;
    const schemaV5 = row.generation.generator.endsWith('/5') || row.generation.generator.endsWith('/6');
    assert.equal(row.generation.answer_normalization, qaMap ? 'squad-token-map/1' :
      schemaV5 ? 'json-string-record/1' : updatedCollection ? 'source-exact/1' : undefined);
    assert.equal(item.answer_format.includes('SQuAD-style canonical form'), false,
      `${row.id}/${path}: task prompts must leave the answer in natural wording`);
    assert.equal(source.outcome.label, 'gold');
    if (source.family === 'qa_extractive' || source.family === 'qa_multihop')
      assert(evidence.includes(source.target.value), `${row.id}/${path}: extractive label must be an exact evidence substring`);
    if (source.target.kind === 'choice')
      assert(new RegExp(`(?:^|\\n)\\s*\\(${source.target.value}\\)\\s`).test(question),
        `${row.id}/${path}: choice label must appear as a shown option`);
    if (source.family === 'table_qa_stored')
      assert(!/[;,]/.test(source.target.value), `${row.id}/${path}: ambiguous unordered multi-value table labels are excluded`);
    assert(!prompts.has(question), `${row.id}: duplicate questions in one collection`); prompts.add(question);
    assert(!seenIds.has(source.id), `source record reused by two tasks: ${source.id}`); seenIds.add(source.id);
  }
  for (const group of groups) {
    assert(!seenGroups.has(group), `source group reused by two task variants: ${group}`);
    seenGroups.add(group);
  }
  assert.equal(row.generation.independent_world_credit, 0);
  assert.equal(row.generation.training_admission, false);
  assert.equal(row.generation.provider_calls, 0);
}
assert(cases.length > 0, 'at least one source-derived task variant is required');
const answerModes = cases.map(row => row.generation.answer_payload_mode ?? 'crisp');
assert(answerModes.every(mode => ['crisp', 'soft'].includes(mode)), 'every task needs an explicit supported answer mode');
for (const row of cases) {
  const mode = row.generation.answer_payload_mode ?? 'crisp';
  const code = row.curriculum.reference.root.find(call => call[0] === 'eval')?.[1]?.code ?? '';
  assert(code.includes(mode === 'soft' ? 'nl<Neuralese<string>>' : 'nl<string>'), `${row.id}: inline result mode differs from source metadata`);
  assert(code.includes(mode === 'soft' ? 'String(answer)' : 'const answerText = answer'), `${row.id}: answer handling differs from source mode`);
  if (['/4', '/5', '/6'].some(version => row.generation.generator.endsWith(version))) {
    const task = JSON.parse(row.semantics.folder_files['task.json']);
    assert.equal(task.output_file, 'answers.json', `${row.id}: output filename must be a distinct path field`);
    assert.equal(typeof task.output_contract, 'string', `${row.id}: output instructions must be a separate contract field`);
    assert(!Object.hasOwn(task, 'output'), `${row.id}: do not overload output with prose that can be mistaken for a path`);
    assert(code.includes('task.output_file') && code.includes('folder.file(outputFile)'), `${row.id}: root must use the declared output path`);
    const rootInstructions = row.semantics.files[row.semantics.root];
    assert(rootInstructions.includes('task.output_contract'), `${row.id}: root guidance must name the output contract field`);
    if (row.generation.generator.endsWith('/5') || row.generation.generator.endsWith('/6')) {
      assert.equal(task.answer_mode, mode, `${row.id}: task answer_mode must match the declared child-output mode`);
      assert(task.instruction.includes('Neuralese<string>') && task.instruction.includes('String(answer)') &&
        task.instruction.includes('nl<string>'), `${row.id}: task must explain both crisp and soft answer handling`);
      assert(rootInstructions.includes('task.instruction'), `${row.id}: root must direct the model to the answer-mode guidance`);
      assert.equal(row.semantics.files_oracle.compare, ['qa_extractive', 'qa_multihop'].includes(row.dataset) ?
        'qa-string-map' : 'json-string-record', `${row.id}: answer file grading must ignore JSON whitespace only`);
    }
  }
  const softChildren = row.curriculum.reference.children.filter(child => child.soft_output);
  assert.equal(softChildren.length, mode === 'soft' ? row.curriculum.reference.children.length : 0,
    `${row.id}: every child output must follow its declared mode`);
}
const sourceProofPath = resolve(dirname(sourcePath), 'source-proof.json');
const sourceProof = JSON.parse(await readFile(sourceProofPath, 'utf8'));
assert.equal(sourceProof.source_cases_sha256, sha(bytes));
assert.equal(sourceProof.task_variants, cases.length);
assert.equal(sourceProof.answer_mode_counts?.crisp ?? cases.length, answerModes.filter(mode => mode === 'crisp').length);
assert.equal(sourceProof.answer_mode_counts?.soft ?? 0, answerModes.filter(mode => mode === 'soft').length);
const report = { schema: 's1-question-collection-independent-verification/2',
  source_cases_sha256: sha(bytes), source_records_verified: sourceRows.size, task_variants_verified: cases.length,
  answer_mode_counts: { crisp: answerModes.filter(mode => mode === 'crisp').length,
    soft: answerModes.filter(mode => mode === 'soft').length },
  original_groups_verified: seenGroups.size, split_counts: Object.fromEntries([...new Set(cases.map(row => row.split))]
    .map(split => [split, cases.filter(row => row.split === split).length])),
  source_file_sha256: Object.fromEntries(filePins), source_manifest_sha256: sha(manifestBytes),
  exact_source_questions_and_evidence: true, original_source_targets_retained_with_family_scoped_squad_map_grader: true, no_gold_payload_fields: true,
  no_duplicate_selected_source_record_or_group: true, independent_world_credit: 0,
  provider_calls: 0, teacher_observations: 0, training_admission: false, semantic_admission: false };
await writeFile(outPath, JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({ report: outPath, ...report }, null, 2));
