#!/usr/bin/env node
import { createHash, randomUUID } from 'node:crypto';
import ts from 'typescript';
import { mkdir, readFile, rename, unlink, writeFile, link } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { digest, readJsonl } from './common.mjs';

const hashText = value => createHash('sha256').update(value).digest('hex');

/** Convert normalized source records without upgrading unverified code into evidence. */
export function compileCurriculum(tasks) {
  return tasks.map(task => {
    const common = { version: 'natlang.code_curriculum/1', id: `curriculum:${digest(task.id).slice(0, 24)}`,
      source_task_id: task.id, group_id: task.group_id, source: task.source, language: task.language,
      instruction: task.instruction, stage: task.stage ?? 'source', verification: task.verification };
    if (task.verification?.status === 'rejected') return { ...common, kind: 'rejected_inventory', target: null,
      evidence: 'rejected_source_excluded_from_training' };
    if (task.generation?.generator === 'natlang.code_curriculum/1') return { ...common, kind: 'synthetic_native_replay', target: 'native_eval_replay',
      function: task.function, cases: task.cases, evidence: 'generated_reference_outputs_require_native_replay_verification' };
    if (task.observation?.kind === 'source_derived_observations') return { ...common, kind: 'source_observed_replay', target: 'native_eval_replay',
      function: task.function, cases: task.cases, evidence: 'observed_from_original_source_not_upstream_tests; require_native_replay' };
    if (task.kind === 'tool_calls') return { ...common, kind: 'external_api_stub', target: 'schema_call_proposal',
      input: { schemas: task.raw?.tools ?? [], proposed_calls: task.raw?.calls ?? [] }, expected: null,
      evidence: 'schema_ordered_call_only', note: 'External API calls are proposals; no tool execution or response is supplied.' };
    if (task.function && Array.isArray(task.cases) && task.cases.some(c => c.outcome === 'return' && c.portable !== false))
      return { ...common, kind: 'captured_function_replay', target: 'native_eval_replay', function: task.function,
        cases: task.cases.filter(c => c.outcome === 'return' && c.portable !== false),
        evidence: 'captured_return_cases_require_native_replay' };
    if (task.function?.source || task.function?.body) return { ...common, kind: 'instruction_code_proposal',
      target: 'code_generation', proposal: task.function.source ?? task.function.body,
      evidence: 'source_candidate_unexecuted', completion_status: 'unverified' };
    return { ...common, kind: 'instruction_only', target: 'code_generation', proposal: null,
      evidence: 'no_executable_source_or_capture' };
  });
}

// Seed/index PRNG makes every task independently reproducible and resumable by index.
function random(seed, index) {
  let state = Buffer.from(hashText(`${seed}:${index}:code-curriculum/1`).slice(0, 8), 'hex').readUInt32BE(0) || 0x9e3779b9;
  return () => { state ^= state << 13; state ^= state >>> 17; state ^= state << 5; return (state >>> 0) / 0x1_0000_0000; };
}
const pick = (rng, values) => values[Math.floor(rng() * values.length)];
const stableSplit = group => Number.parseInt(digest(group).slice(0, 8), 16) % 100 < 5 ? 'test' : 'train';

/** Small, auditable algorithmic tasks whose argument/result boundaries fit native replay. */
export function syntheticCodeTasks(seed = 0, startIndex = 0, count = 100) {
  if (![seed, startIndex, count].every(Number.isSafeInteger) || startIndex < 0 || count < 0)
    throw new Error('seed, start index, and count must be safe integers; index and count must be nonnegative');
  const templates = [
    { name: 'stable_unique', param: 'items', type: 'string[]', out: 'string[]', instruction: 'Remove duplicate strings from items while preserving their first occurrence order.', body: '{ return items.filter((value, index) => items.indexOf(value) === index); }', make: r => Array.from({length: 5 + Math.floor(r()*6)}, () => pick(r,['amber','blue','cedar','dune'])) },
    { name: 'prefix_sums', param: 'numbers', type: 'number[]', out: 'number[]', instruction: 'Return the running total after each value in numbers.', body: '{ let total = 0; return numbers.map(value => total += value); }', make: r => Array.from({length: 3 + Math.floor(r()*7)}, () => Math.floor(r()*21)-10) },
    { name: 'count_words', param: 'text', type: 'string', out: 'number', instruction: 'Count whitespace-separated words in text; an empty or whitespace-only string has zero words.', body: '{ const trimmed = text.trim(); return trimmed ? trimmed.split(/\\s+/).length : 0; }', make: r => pick(r,['','  ','red blue','one\t two  three','north south east west']) },
    { name: 'group_count', param: 'items', type: 'string[]', out: 'string[][]', instruction: 'Return each distinct string paired with its occurrence count, ordered by first appearance.', body: '{ const pairs = []; for (const value of items) { const pair = pairs.find(item => item[0] === value); if (pair) pair[1] = String(Number(pair[1]) + 1); else pairs.push([value, "1"]); } return pairs; }', make: r => Array.from({length: 4 + Math.floor(r()*6)}, () => pick(r,['oak','elm','ash'])) },
    { name: 'record_total', param: 'entries', type: 'string[][]', out: 'number', instruction: 'Sum the numeric amount stored in position 1 of each [name, amount] entry.', body: '{ return entries.reduce((total, entry) => total + Number(entry[1]), 0); }', make: r => Array.from({length: 2 + Math.floor(r()*6)}, (_,i) => [`item${i}`, String(Math.floor(r()*20))]) },
    { name: 'longest_word', param: 'words', type: 'string[]', out: 'string', instruction: 'Return the longest word, keeping the earliest word when lengths tie; return an empty string for no words.', body: '{ return words.reduce((best, word) => word.length > best.length ? word : best, ""); }', make: r => Array.from({length: Math.floor(r()*7)}, () => pick(r,['a','elm','stone','cypress','moon'])) },
  ];
  const rows = [];
  for (let index = startIndex; index < startIndex + count; index++) {
    const rng = random(seed, index), t = templates[index % templates.length], args = t.make(rng);
    // Second case broadens the inferred input/output type and covers empty/singleton edges.
    const edge = t.name === 'count_words' ? '' : t.name === 'record_total' ? [] : t.name === 'longest_word' ? [] : [];
    const expected = value => {
      if (t.name === 'stable_unique') return value.filter((item, i) => value.indexOf(item) === i);
      if (t.name === 'prefix_sums') { let total = 0; return value.map(item => total += item); }
      if (t.name === 'count_words') return value.trim() ? value.trim().split(/\s+/).length : 0;
      if (t.name === 'group_count') { const counts = new Map(); for (const item of value) counts.set(item, (counts.get(item) ?? 0) + 1); return [...counts].map(([key, n]) => [key, String(n)]); }
      if (t.name === 'record_total') return value.reduce((total, entry) => total + Number(entry[1]), 0);
      return value.reduce((best, word) => word.length > best.length ? word : best, '');
    };
    const cases = [args, edge].map(value => ({ args: [value], expected: expected(value), outcome: 'return', portable: true }));
    // All examples of the same algorithm share a split group, preventing case leakage.
    const group = `natlang-synthetic-code:${t.name}`;
    rows.push({ version: 'natlang.code_task/1', id: `${group}:${seed}:${index}`, group_id: group, kind: 'function', language: 'javascript',
      instruction: t.instruction, source: { name: 'natlang-synthetic-code', revision: 'code-curriculum/1', path: 'generated', license: 'project-generated', split: stableSplit(group) },
      function: { name: t.name, parameters: [{ name: t.param }], body: t.body, source: `function ${t.name}(${t.param}) ${t.body}` },
      cases, verification: { status: 'generated_candidate', reasons: ['deterministic_reference_outputs_require_native_replay_verification'] },
      generation: { generator: 'natlang.code_curriculum/1', seed, index, difficulty: ['basic','intermediate','intermediate','advanced','advanced','basic'][index % templates.length], boundary: t.type, output: t.out } });
  }
  return rows;
}

export async function codeProposalTurns(tasks) {
  return tasks.flatMap(task => {
    if (task.verification?.status === 'rejected' || !['javascript','typescript'].includes(task.language) || !task.instruction?.trim()) return [];
    const proposal = task.function?.source ?? task.function?.body;
    if (typeof proposal !== 'string' || !proposal.trim() || task.kind === 'tool_calls') return [];
    const sourceFile = ts.createSourceFile(task.language === 'typescript' ? 'proposal.ts' : 'proposal.js', proposal,
      ts.ScriptTarget.Latest, true, task.language === 'typescript' ? ts.ScriptKind.TS : ts.ScriptKind.JS);
    if (sourceFile.parseDiagnostics.length) return [];
    const group = task.group_id ?? task.id;
    const explicitSplit = task.split ?? task.source?.split ?? task.source?.upstream_split;
    const split = ['test','validation','valid','dev'].includes(explicitSplit) ? 'test' : ['train'].includes(explicitSplit) ? 'train' : stableSplit(group);
    return [{ version: 'natlang.teacher_training_turn.native/1', id: `code-proposal:${digest(task.id).slice(0,24)}`,
      program_id: group, source_groups: [group], family: 'general_code_proposal', skill: 'code_generation', split,
      source: task.source, license: task.source?.license ?? null, execution_verified: false,
      messages: [{ role: 'system', content: 'Write the requested source code. Return code only.' },
        { role: 'user', content: task.instruction }], tools: [],
      target: { role: 'assistant', content: proposal },
      training_admission: { approved: true, reason: 'unverified-direct-code-proposal' }, evidence: 'source_code_not_execution_verified' }];
  });
}

async function atomicWrite(path, contents) {
  const target = resolve(path); await mkdir(dirname(target), { recursive: true });
  const staging = `${target}.building-${randomUUID()}`;
  try { await writeFile(staging, contents, { flag: 'wx' }); await rename(staging, target); }
  finally { await unlink(staging).catch(() => {}); }
}
async function atomicCacheWrite(path, contents) {
  const target = resolve(path), staging = `${target}.building-${randomUUID()}`;
  try {
    await writeFile(staging, contents, { flag: 'wx' });
    try { await link(staging, target); } catch (error) { if (error.code !== 'EEXIST') throw error; }
  } finally { await unlink(staging).catch(() => {}); }
}
const jsonl = rows => rows.map(row => JSON.stringify(row)).join('\n') + (rows.length ? '\n' : '');
let interrupted = false;
const requestCheckpoint = signal => { interrupted = true; process.stderr.write(`\n${signal}: checkpointing after the current replay case\n`); };

async function runtimeFingerprint() {
  const paths = ['dist/environment.js','dist/application-packages.js','dist/native/agent.js','dist/native/runtime.js',
    'dist/native/types.js','dist/native/values.js','dist/native/prompt.js','dist/teacher/native-materializer.js'];
  const files = {};
  for (const path of paths) files[path] = hashText(await readFile(new URL(`../../${path}`, import.meta.url)));
  return hashText(JSON.stringify({ node:process.version, files }));
}

async function main() {
  const args = process.argv.slice(2), value = name => { const i = args.indexOf(name); return i < 0 ? undefined : args[i+1]; };
  if (args.includes('--help') || args.includes('-h')) { process.stdout.write('Usage: node scripts/code-corpus/curriculum.mjs --out DIR [--input TASKS.jsonl ...] [--seed N --start-index N --synthetic N] [--replay-synthetic] [--replay-candidates]\nOutputs general-code.jsonl and code-proposals.jsonl (unverified general code SFT), teacher-programs.jsonl (projected natlang.program/1 IR), replay-trajectories.jsonl and verified-turns.jsonl (only with replay). Interrupted replay checkpoints with exit 75; rerun the same command to resume.\n'); return; }
  const out = value('--out'); if (!out) throw new Error('--out is required');
  const inputs = args.flatMap((arg, i) => arg === '--input' ? [args[i+1]] : []).filter(Boolean);
  const seed = Number(value('--seed') ?? 0), start = Number(value('--start-index') ?? 0), count = Number(value('--synthetic') ?? 0);
  const sourceTasks = (await Promise.all(inputs.map(path => readJsonl(path)))).flat();
  const generated = syntheticCodeTasks(seed, start, count);
  const outPath = resolve(out); await mkdir(outPath, { recursive: true });
  const replaySynthetic = args.includes('--replay-synthetic'), replayCandidates = args.includes('--replay-candidates');
  const replayEnabled = replaySynthetic || replayCandidates;
  const inputHashes = await Promise.all(inputs.map(async path => [resolve(path), hashText(await readFile(path))]));
  const curriculumHash = hashText(await readFile(new URL('./curriculum.mjs', import.meta.url)));
  const runtimeHash = replayEnabled ? await runtimeFingerprint() : null;
  const config = { generator: 'natlang.code_curriculum/1', curriculum_sha256: curriculumHash, seed, start_index: start, synthetic_count: count,
    inputs: inputHashes, replay_synthetic: replaySynthetic, replay_candidates: replayCandidates, runtime_sha256: runtimeHash,
    tools_sha256: digest([]), prompt_sha256: hashText('Write the requested source code. Return code only.') };
  const manifestPath = resolve(outPath, 'manifest.json');
  try {
    const previous = JSON.parse(await readFile(manifestPath, 'utf8'));
    if (JSON.stringify(previous.config) !== JSON.stringify(config)) throw new Error('output directory belongs to a different immutable curriculum configuration');
  } catch (error) { if (error.code !== 'ENOENT') throw error; }
  await atomicWrite(manifestPath, `${JSON.stringify({ version: 'natlang.code_curriculum_manifest/1', config, status: 'in_progress' }, null, 2)}\n`);
  process.once('SIGINT', () => requestCheckpoint('SIGINT'));
  process.once('SIGTERM', () => requestCheckpoint('SIGTERM'));
  await atomicWrite(resolve(out, 'synthetic-code-tasks.jsonl'), jsonl(generated));
  const curriculum = compileCurriculum([...sourceTasks, ...generated]);
  const proposalRows = curriculum.filter(row => ['instruction_code_proposal','external_api_stub','instruction_only','rejected_inventory'].includes(row.kind));
  await atomicWrite(resolve(out, 'general-code.jsonl'), jsonl(proposalRows));
  const allTasks = [...sourceTasks, ...generated];
  const nativeProposals = await codeProposalTurns(allTasks);
  await atomicWrite(resolve(out, 'code-proposals.jsonl'), jsonl(nativeProposals));
  await atomicWrite(resolve(out, 'curriculum.jsonl'), jsonl(curriculum));
  const { project, replayIsolated, materializeCorpus } = await import('./replay.mjs');
  const programs = [], projectionRejected = [];
  for (const task of allTasks) {
    if (task.verification?.status === 'rejected') continue;
    const cases = (task.cases ?? []).filter(item => item.outcome === 'return' && item.portable !== false);
    for (let index = 0; index < cases.length; index++) {
      try {
        const program = project(task, index).program;
        const explicit = task.split ?? task.source?.split ?? task.source?.upstream_split;
        program.split = ['test','validation','valid','dev'].includes(explicit) ? 'test' : ['train'].includes(explicit) ? 'train' : stableSplit(task.group_id);
        const curriculumEvidence = task.generation ? 'generated_reference_unverified_until_native_replay' : task.observation ? 'source_observation_unverified_until_native_replay_not_upstream_test' : 'capture_projected_unverified_until_native_replay';
        programs.push({ ...program, curriculum_evidence: curriculumEvidence });
      } catch (error) { projectionRejected.push({ task_id: task.id, index, error: String(error) }); }
    }
  }
  await atomicWrite(resolve(out, 'teacher-programs.jsonl'), jsonl(programs));
  if (replayEnabled) {
    const cache = resolve(out, 'replay-cache'); await mkdir(cache, { recursive: true });
    const trajectories = [];
    const replayTasks = [
      ...(replayCandidates ? sourceTasks.filter(task => task.verification?.status !== 'rejected' && task.kind !== 'tool_calls' && (task.cases ?? []).some(item=>item.outcome==='return'&&item.portable!==false)).map(task=>({...task,cases:task.cases.filter(item=>item.outcome==='return'&&item.portable!==false)})) : []),
      ...(replaySynthetic ? generated : []),
    ];
    for (const task of replayTasks) {
      for (let i = 0; i < (task.cases ?? []).length; i++) {
        if (interrupted) break;
        const cachePath = resolve(cache, `${digest([runtimeHash, config.tools_sha256, task.id, task.function, task.cases[i]]).slice(0,40)}.json`);
        let row;
        try { row = JSON.parse(await readFile(cachePath, 'utf8')); }
        catch (error) {
          if (error.code !== 'ENOENT') throw error;
          row = await replayIsolated(task, i);
          if (task.observation) row.provenance.source_observation = structuredClone(task.observation);
          await atomicCacheWrite(cachePath, `${JSON.stringify(row)}\n`);
        }
        trajectories.push(row);
      }
      if (interrupted) break;
    }
    const materialized = await materializeCorpus(trajectories);
    await atomicWrite(resolve(out, 'replay-trajectories.jsonl'), jsonl(trajectories));
    await atomicWrite(resolve(out, 'verified-turns.jsonl'), jsonl(materialized.turns));
    await atomicWrite(resolve(out, 'projection-rejected.jsonl'), jsonl(projectionRejected));
    const status = interrupted ? 'interrupted' : 'complete';
    await atomicWrite(manifestPath, `${JSON.stringify({ version: 'natlang.code_curriculum_manifest/1', config, status,
      synthetic_tasks: generated.length, source_tasks: sourceTasks.length, curriculum_tasks: allTasks.length,
      curriculum_by_kind: Object.fromEntries([...new Set(curriculum.map(row => row.kind))].sort().map(kind => [kind,curriculum.filter(row => row.kind === kind).length])),
      projected_programs: programs.length, general_code_proposal_turns: nativeProposals.length, trajectories: trajectories.length,
      accepted_trajectories: trajectories.filter(row => row.outcome.accepted).length,
      accepted_source_observation_replays: trajectories.filter(row => row.provenance.source_observation && row.outcome.accepted).length,
      verified_turns: materialized.turns.length,
      outputs: ['general-code.jsonl','code-proposals.jsonl','teacher-programs.jsonl','replay-trajectories.jsonl','verified-turns.jsonl','projection-rejected.jsonl'],
      replay_cache: 'replay-cache/' }, null, 2)}\n`);
    if (interrupted) process.exitCode = 75;
    return;
  }
  const manifest = { version: 'natlang.code_curriculum_manifest/1', config, status: 'complete', synthetic_tasks: generated.length,
    source_tasks: sourceTasks.length, curriculum_tasks: allTasks.length,
    curriculum_by_kind: Object.fromEntries([...new Set(curriculum.map(row => row.kind))].sort().map(kind => [kind,curriculum.filter(row => row.kind === kind).length])),
    projected_programs: programs.length, general_code_proposal_turns: nativeProposals.length, projection_rejected: projectionRejected.length,
    outputs: ['general-code.jsonl','code-proposals.jsonl','teacher-programs.jsonl','projection-rejected.jsonl'] };
  await atomicWrite(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
  process.stdout.write(`${manifest.curriculum_tasks} curriculum tasks -> ${outPath}\n`);
}
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) main().catch(error => { process.stderr.write(`${error}\n`); process.exitCode = 1; });
