#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { referenceDriver } from '../../dist/teacher/curriculum.js';
import { defaultToolSurfaceHash, executeProgram } from '../../dist/teacher/collector.js';
import { MemoryNeuraleseStore, StandInNeuralesePort, hashingEmbedder } from '../../dist/native/neuralese-store.js';
import { TOOLS_PROMPT } from '../../dist/native/prompt.js';

const { values } = parseArgs({ options: {
  source: { type: 'string' },
  out: { type: 'string' },
}});
if (!values.source || !values.out) throw new Error('usage: node prove-semantic-source-worlds-v13.mjs --source SOURCE_JSONL --out FRESH_PROOF_JSON');

const sourcePath = resolve(values.source);
const outPath = resolve(values.out);
const canonical = value => Array.isArray(value) ? `[${value.map(canonical).join(',')}]` :
  value && typeof value === 'object' ? `{${Object.keys(value).sort().map(key => `${JSON.stringify(key)}:${canonical(value[key])}`).join(',')}}` : JSON.stringify(value);
const sourceBytes = await readFile(sourcePath);
const sourceSha = createHash('sha256').update(sourceBytes).digest('hex');
const rows = sourceBytes.toString('utf8').trimEnd().split('\n').map(line => JSON.parse(line));
if (rows.length !== 24) throw new Error(`expected 24 source records, got ${rows.length}`);
const toolSurfaceSha256 = await defaultToolSurfaceHash();
const cases = [];

for (let index = 0; index < rows.length; index++) {
  const record = rows[index];
  const store = new MemoryNeuraleseStore();
  const port = new StandInNeuralesePort(store, hashingEmbedder(8), 8, 'nd:natlang@1');
  const driver = Object.assign(referenceDriver(record), { neuralese: true });
  const result = await executeProgram(record, driver, {
    rootSeed: 909,
    runId: `v13-cpu-reference-${String(index).padStart(2, '0')}`,
    systemPrompt: TOOLS_PROMPT,
    contextTokens: 65_536,
    maxTurns: 60,
    toolSurfaceSha256,
    collectionRole: 'reference',
    neuralese: { store, port },
  });

  const ledger = result.outcome.invocation_ledger ?? [];
  const root = ledger.find(entry => entry.parent_invocation_id === null);
  if (!root) throw new Error(`${record.id}: runtime returned no root invocation`);
  const children = ledger.filter(entry => entry.parent_invocation_id === root.invocation_id);
  const events = result.outcome.action_ledger ?? [];
  const childReads = children.map(child => {
    const read = events.find(event => event.call_id === child.invocation_id && event.name === 'eval' &&
      (String(event.arguments?.code).includes('file.readText()') || String(event.arguments?.code).includes('packet.readText()')));
    if (!read || read.outcome !== 'ok') throw new Error(`${record.id}: child ${child.invocation_id} did not read its supplied evidence FileHandle`);
    const evidence = record.curriculum.iterate === 'required' ?
      [{ path: 'packet.md', text: record.semantics.folder_files['packet.md'] }] :
      Object.entries(record.semantics.folder_files).filter(([path]) => path.startsWith('records/')).map(([path, text]) => ({ path, text }));
    const matched = evidence.filter(item => String(read.result_text ?? '').includes(item.text));
    if (matched.length !== 1) throw new Error(`${record.id}: child evidence read did not show exactly one complete source file`);
    return { invocation_id: child.invocation_id, evidence_path: matched[0].path, evidence_text_matches_source: true,
      host_result: child.host_result?.value, read_result_sha256: createHash('sha256').update(String(read.result_text ?? '')).digest('hex') };
  });
  if (result.outcome.status !== 'done' || result.outcome.accepted !== true)
    throw new Error(`${record.id}: runtime reference did not complete cleanly: ${JSON.stringify(result.outcome.rejection_reasons)}`);
  if (canonical(result.outcome.value) !== canonical(record.semantics.expected))
    throw new Error(`${record.id}: runtime reference output differs from source expected value`);
  if (canonical(result.outcome.files) !== canonical(record.semantics.expected_files))
    throw new Error(`${record.id}: runtime reference files differ from expected_files`);

  cases.push({
    id: record.id,
    source_group: record.source_groups[0],
    split: record.split,
    kind: record.curriculum.iterate === 'required' ? 'iterateOn' : 'FileHandle',
    status: result.outcome.status,
    accepted_by_runtime_oracles: result.outcome.accepted,
    value_matches_source_expected: true,
    files_match_expected: true,
    child_invocations: children.length,
    child_evidence_reads: childReads,
  });
}

const proof = {
  schema: 'natlang.neuralese-semantic-source-v13.cpu-runtime-reference-proof/1',
  source_path: sourcePath,
  source_sha256: sourceSha,
  runtime: 'compiled shared TypeScript host runtime; CPU-only referenceDriver and StandInNeuralesePort',
  model_calls: 0,
  provider_calls: 0,
  neuralese_model_calls: 0,
  teacher_trajectories: 0,
  admission_granted: false,
  actual_runtime_cases: cases.length,
  clean_runtime_completions: cases.filter(item => item.status === 'done' && item.accepted_by_runtime_oracles).length,
  child_filehandle_reads: cases.reduce((sum, item) => sum + item.child_evidence_reads.length, 0),
  proof_limit: 'Confirms compiled-runtime execution, supplied-handle evidence reads, and scripted reference output/file contracts. The deterministic referenceDriver supplies scripted child decisions; this does not establish semantic truth or teacher behavior.',
  cases,
};

await mkdir(dirname(outPath), { recursive: true });
try {
  await writeFile(outPath, JSON.stringify(proof, null, 2) + '\n', { flag: 'wx' });
} catch (error) {
  if (error.code === 'EEXIST') throw new Error(`refusing to overwrite existing CPU proof: ${outPath}`);
  throw error;
}
console.log(JSON.stringify({ out: outPath, source_sha256: sourceSha, cases: cases.length,
  clean: proof.clean_runtime_completions, child_filehandle_reads: proof.child_filehandle_reads,
  model_calls: 0, provider_calls: 0, admission_granted: false }, null, 2));
