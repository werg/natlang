#!/usr/bin/env node
import { createHash, randomUUID } from 'node:crypto';
import ts from 'typescript';
import { mkdir, readFile, rename, unlink, writeFile, link } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { digest, readJsonl } from './common.mjs';
import { DEFAULT_MAX_FAMILY_REPETITIONS, makeSyntheticCases, SYNTHETIC_CODE_FAMILIES, SYNTHETIC_CODE_GENERATOR } from './synthetic-families.mjs';
import { CODE_ONLY_GENERATOR, syntheticCodeOnlyTasks } from './code-only-families.mjs';

const hashText = value => createHash('sha256').update(value).digest('hex');

/** Convert normalized source records without upgrading unverified code into evidence. */
export function compileCurriculum(tasks) {
  return tasks.map(task => {
    const common = { version: 'natlang.code_curriculum/1', id: `curriculum:${digest(task.id).slice(0, 24)}`,
      source_task_id: task.id, group_id: task.group_id, source: task.source, language: task.language,
      instruction: task.instruction, stage: task.stage ?? 'source', verification: task.verification,
      behavioral_evidence: task.behavioral_evidence, generation: task.generation };
    if (task.verification?.status === 'rejected') return { ...common, kind: 'rejected_inventory', target: null,
      evidence: 'rejected_source_excluded_from_training' };
    if (task.generation?.generator?.startsWith('natlang.code_curriculum/')) return { ...common, kind: 'synthetic_native_replay', target: 'native_eval_replay',
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
  let state = Buffer.from(hashText(`${seed}:${index}:code-curriculum/2`).slice(0, 8), 'hex').readUInt32BE(0) || 0x9e3779b9;
  return () => { state ^= state << 13; state ^= state >>> 17; state ^= state << 5; return (state >>> 0) / 0x1_0000_0000; };
}
const stableSplit = group => Number.parseInt(digest(group).slice(0, 8), 16) % 100 < 5 ? 'test' : 'train';

/** Small, auditable algorithmic tasks whose argument/result boundaries fit native replay. */
export function syntheticCodeTasks(seed = 0, startIndex = 0, count = 100, maxFamilyRepetitions = DEFAULT_MAX_FAMILY_REPETITIONS) {
  if (![seed, startIndex, count, maxFamilyRepetitions].every(Number.isSafeInteger) || startIndex < 0 || count < 0 || maxFamilyRepetitions < 1)
    throw new Error('seed, start index, count, and family repetition cap must be safe integers; index/count must be nonnegative and cap positive');
  const emittedCount = Math.min(count, Math.max(0, SYNTHETIC_CODE_FAMILIES.length * maxFamilyRepetitions - startIndex));
  const rows = [];
  for (let index = startIndex; index < startIndex + emittedCount; index++) {
    const rng = random(seed, index), t = SYNTHETIC_CODE_FAMILIES[index % SYNTHETIC_CODE_FAMILIES.length], args = t.make(rng);
    const cases = makeSyntheticCases(t, args);
    // All examples of the same algorithm share a split group, preventing case leakage.
    const group = `natlang-synthetic-code:${t.name}`;
    rows.push({ version: 'natlang.code_task/1', id: `${SYNTHETIC_CODE_GENERATOR}:${group}:${seed}:${index}`, group_id: group, kind: 'function', language: 'javascript',
      instruction: t.instruction, source: { name: 'natlang-synthetic-code', revision: SYNTHETIC_CODE_GENERATOR, path: 'generated', license: 'project-generated', split: stableSplit(group) },
      function: { name: t.name, parameters: [{ name: t.param, type: t.type }], return_type: t.out,
        body: t.body, source: `function ${t.name}(${t.param}) ${t.body}` },
      cases, verification: { status: 'generated_candidate', reasons: ['deterministic_reference_outputs_require_native_replay_verification'] },
      generation: { generator: SYNTHETIC_CODE_GENERATOR, seed, index, family: t.name, difficulty: t.difficulty, boundary: t.type, output: t.out },
      behavioral_evidence: { kind:'synthetic_reference_and_properties', status:'reference_checks_passed_pending_native_replay',
        reference:'family-specific executable reference; algorithmic independence is not claimed',
        property_checks:cases.map((item,case_index)=>({ case_index, checks:item.property_checks })) } });
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
      program_id: group, source_groups: [group], family: task.generation?.generator === CODE_ONLY_GENERATOR ? task.generation.family : 'general_code_proposal', skill: 'code_generation', split,
      source: task.source, license: task.source?.license ?? null, execution_verified: false,
      generation:task.generation, implementation_sha256:digest(task.function), syntax_checked:true,
      messages: [{ role: 'system', content: 'Write the requested source code. Return code only.' },
        { role: 'user', content: task.instruction }], tools: [],
      target: { role: 'assistant', content: proposal },
      training_admission: { approved: true, reason: 'unverified-direct-code-proposal' }, evidence: 'source_code_not_execution_verified',
      behavioral_evidence: task.behavioral_evidence ?? { kind:'syntax_only', status:'not_execution_verified', properties:['source parses as JavaScript or TypeScript'] } }];
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
const requestCheckpoint = signal => { if (interrupted) return; interrupted = true; process.stderr.write(`\n${signal}: checkpointing after the current replay case\n`); };

async function runtimeFingerprint() {
  const paths = ['dist/environment.js','dist/application-packages.js','dist/native/agent.js','dist/native/runtime.js',
    'dist/native/types.js','dist/native/values.js','dist/native/prompt.js','dist/teacher/native-materializer.js'];
  const files = {};
  for (const path of paths) files[path] = hashText(await readFile(new URL(`../../${path}`, import.meta.url)));
  return hashText(JSON.stringify({ node:process.version, files }));
}
async function toolFingerprint() {
  const paths=['dist/native/agent.js','dist/native/prompt.js','dist/application-packages.js'];
  const files={}; for(const path of paths) files[path]=hashText(await readFile(new URL(`../../${path}`,import.meta.url)));
  return hashText(JSON.stringify(files));
}
async function outputHashes(output,names) {
  return Object.fromEntries(await Promise.all(names.map(async name=>[name,hashText(await readFile(resolve(output,name)))])));
}
async function validateOutputHashes(output,hashes) {
  for(const [name,expected] of Object.entries(hashes??{})) {
    const actual=hashText(await readFile(resolve(output,name)));
    if(actual!==expected) throw new Error(`completed output integrity check failed for ${name}; refusing silent repair`);
  }
}

async function main() {
  const args = process.argv.slice(2), value = name => { const i = args.indexOf(name); return i < 0 ? undefined : args[i+1]; };
  if (args.includes('--help') || args.includes('-h')) { process.stdout.write('Usage: node scripts/code-corpus/curriculum.mjs --out DIR [--input TASKS.jsonl ...] [--seed N --start-index N --synthetic N --max-family-repetitions N] [--replay-synthetic] [--replay-candidates]\nA positive --synthetic count also adds up to seven unique syntax-only code proposals for object, async, error, and Node builtin import tasks. Outputs general-code.jsonl and code-proposals.jsonl, teacher-programs.jsonl, and replay outputs only for replayable tasks. Interrupted replay checkpoints with exit 75; rerun the same command to resume.\n'); return; }
  const out = value('--out'); if (!out) throw new Error('--out is required');
  const inputs = args.flatMap((arg, i) => arg === '--input' ? [args[i+1]] : []).filter(Boolean);
  const seed = Number(value('--seed') ?? 0), start = Number(value('--start-index') ?? 0), count = Number(value('--synthetic') ?? 0);
  const maxFamilyRepetitions = Number(value('--max-family-repetitions') ?? DEFAULT_MAX_FAMILY_REPETITIONS);
  const sourceTasks = (await Promise.all(inputs.map(path => readJsonl(path)))).flat();
  const generated = syntheticCodeTasks(seed, start, count, maxFamilyRepetitions);
  const generatedCodeOnly = count > 0 ? syntheticCodeOnlyTasks(seed) : [];
  const allGenerated = [...generated, ...generatedCodeOnly];
  const outPath = resolve(out); await mkdir(outPath, { recursive: true });
  const replaySynthetic = args.includes('--replay-synthetic'), replayCandidates = args.includes('--replay-candidates');
  const replayEnabled = replaySynthetic || replayCandidates;
  const inputHashes = await Promise.all(inputs.map(async path => [resolve(path), hashText(await readFile(path))]));
  const curriculumHash = hashText(await readFile(new URL('./curriculum.mjs', import.meta.url)));
  const familiesHash = hashText(await readFile(new URL('./synthetic-families.mjs', import.meta.url)));
  const codeOnlyFamiliesHash = hashText(await readFile(new URL('./code-only-families.mjs', import.meta.url)));
  const runtimeHash = replayEnabled ? await runtimeFingerprint() : null;
  const toolsHash = replayEnabled ? await toolFingerprint() : digest([]);
  // Projection is used even when execution/replay is disabled.
  const replayScriptHash = hashText(await readFile(new URL('./replay.mjs',import.meta.url)));
  const config = { generator: SYNTHETIC_CODE_GENERATOR, curriculum_sha256: curriculumHash, synthetic_families_sha256: familiesHash,
    code_only_generator:CODE_ONLY_GENERATOR, code_only_families_sha256:codeOnlyFamiliesHash, seed, start_index: start,
    synthetic_requested_count: count, synthetic_emitted_count: generated.length, synthetic_capped: generated.length < count,
    code_only_task_count:generatedCodeOnly.length,
    max_family_repetitions: maxFamilyRepetitions,
    inputs: inputHashes, replay_synthetic: replaySynthetic, replay_candidates: replayCandidates,
    replay_sha256: replayScriptHash, cache_version:'natlang.code_curriculum_cache/2', runtime_sha256: runtimeHash,
    tools_sha256: toolsHash, prompt_sha256: hashText('Write the requested source code. Return code only.') };
  const manifestPath = resolve(outPath, 'manifest.json');
  try {
    const previous = JSON.parse(await readFile(manifestPath, 'utf8'));
    if (JSON.stringify(previous.config) !== JSON.stringify(config)) throw new Error('output directory belongs to a different immutable curriculum configuration');
    if (['complete','interrupted'].includes(previous.status) && previous.output_sha256) await validateOutputHashes(outPath,previous.output_sha256);
  } catch (error) { if (error.code !== 'ENOENT') throw error; }
  await atomicWrite(manifestPath, `${JSON.stringify({ version: 'natlang.code_curriculum_manifest/1', config, status: 'in_progress' }, null, 2)}\n`);
  process.on('SIGINT', () => requestCheckpoint('SIGINT'));
  process.on('SIGTERM', () => requestCheckpoint('SIGTERM'));
  await atomicWrite(resolve(out, 'synthetic-code-tasks.jsonl'), jsonl(allGenerated));
  const curriculum = compileCurriculum([...sourceTasks, ...allGenerated]);
  const proposalRows = curriculum.filter(row => ['instruction_code_proposal','external_api_stub','instruction_only','rejected_inventory'].includes(row.kind));
  await atomicWrite(resolve(out, 'general-code.jsonl'), jsonl(proposalRows));
  const allTasks = [...sourceTasks, ...allGenerated];
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
        if (task.generation) {
          program.family = task.generation.family;
          program.generation = structuredClone(task.generation);
          program.behavioral_evidence = structuredClone(task.behavioral_evidence);
        } else if (task.behavioral_evidence) program.behavioral_evidence = structuredClone(task.behavioral_evidence);
        const curriculumEvidence = task.generation ? 'generated_reference_unverified_until_native_replay' : task.observation ? 'source_observation_unverified_until_native_replay_not_upstream_test' : 'capture_projected_unverified_until_native_replay';
        programs.push({ ...program, curriculum_evidence: curriculumEvidence });
      } catch (error) { projectionRejected.push({ task_id: task.id, index, error: String(error) }); }
    }
  }
  await atomicWrite(resolve(out,'projection-rejected.jsonl'),jsonl(projectionRejected));
  await atomicWrite(resolve(out, 'teacher-programs.jsonl'), jsonl(programs));
  if (replayEnabled) {
    const cache = resolve(out, 'replay-cache'); await mkdir(cache, { recursive: true });
    const trajectories = [], replayErrors=[];
    const replayTasks = [
      ...(replayCandidates ? sourceTasks.filter(task => task.verification?.status !== 'rejected' && task.kind !== 'tool_calls' && (task.cases ?? []).some(item=>item.outcome==='return'&&item.portable!==false)).map(task=>({...task,cases:task.cases.filter(item=>item.outcome==='return'&&item.portable!==false)})) : []),
      ...(replaySynthetic ? generated : []),
    ];
    for (const task of replayTasks) {
      for (let i = 0; i < (task.cases ?? []).length; i++) {
        if (interrupted) break;
        const cacheKey=digest([config,task.id,task.function,task.cases[i]]);
        const cachePath = resolve(cache, `${cacheKey.slice(0,40)}.json`);
        let cached;
        try { cached = JSON.parse(await readFile(cachePath, 'utf8')); }
        catch (error) {
          if (error.code !== 'ENOENT') throw error;
          try {
            const row = await replayIsolated(task, i);
            if (task.observation) row.provenance.source_observation = structuredClone(task.observation);
            if (task.behavioral_evidence) row.provenance.behavioral_evidence = structuredClone(task.behavioral_evidence);
            if (task.generation) { row.provenance.family = task.generation.family; row.provenance.generation = structuredClone(task.generation); }
            cached={cache_key:cacheKey,row,payload_sha256:hashText(JSON.stringify(row))};
          } catch (replayError) {
            const errorText=String(replayError);
            cached={cache_key:cacheKey,error:errorText,payload_sha256:hashText(errorText)};
          }
          await atomicCacheWrite(cachePath, `${JSON.stringify(cached)}\n`);
        }
        const payload=cached.row ? JSON.stringify(cached.row) : String(cached.error);
        if(cached.cache_key!==cacheKey || cached.payload_sha256!==hashText(payload))
          throw new Error(`replay cache integrity check failed for ${task.id} case ${i}; refusing silent repair`);
        if(cached.row) trajectories.push(cached.row);
        else replayErrors.push({task_id:task.id,index:i,error:cached.error});
      }
      if (interrupted) break;
    }
    const materialized = await materializeCorpus(trajectories);
    await atomicWrite(resolve(out, 'replay-trajectories.jsonl'), jsonl(trajectories));
    await atomicWrite(resolve(out, 'verified-turns.jsonl'), jsonl(materialized.turns));
    await atomicWrite(resolve(out, 'replay-errors.jsonl'),jsonl(replayErrors));
    const status = interrupted ? 'interrupted' : 'complete';
    const outputs=['synthetic-code-tasks.jsonl','curriculum.jsonl','general-code.jsonl','code-proposals.jsonl','teacher-programs.jsonl',
      'projection-rejected.jsonl','replay-trajectories.jsonl','verified-turns.jsonl','replay-errors.jsonl'];
    const hashes=await outputHashes(outPath,outputs);
    await atomicWrite(manifestPath, `${JSON.stringify({ version: 'natlang.code_curriculum_manifest/1', config, status, output_sha256:hashes,
      synthetic_tasks: generated.length, synthetic_requested_count:count, synthetic_emitted_count:generated.length, synthetic_capped:generated.length<count,
      code_only_tasks:generatedCodeOnly.length,
      source_tasks: sourceTasks.length, curriculum_tasks: allTasks.length,
      curriculum_by_kind: Object.fromEntries([...new Set(curriculum.map(row => row.kind))].sort().map(kind => [kind,curriculum.filter(row => row.kind === kind).length])),
      projected_programs: programs.length, general_code_proposal_turns: nativeProposals.length, trajectories: trajectories.length,
      replay_errors:replayErrors.length,
      accepted_trajectories: trajectories.filter(row => row.outcome.accepted).length,
      accepted_source_observation_replays: trajectories.filter(row => row.provenance.source_observation && row.outcome.accepted).length,
      verified_turns: materialized.turns.length,
      outputs,
      replay_cache: 'replay-cache/' }, null, 2)}\n`);
    if (interrupted) process.exitCode = 75;
    return;
  }
  const manifest = { version: 'natlang.code_curriculum_manifest/1', config, status: 'complete', synthetic_tasks: generated.length,
    synthetic_requested_count:count, synthetic_emitted_count:generated.length, synthetic_capped:generated.length<count,
    code_only_tasks:generatedCodeOnly.length,
    source_tasks: sourceTasks.length, curriculum_tasks: allTasks.length,
    curriculum_by_kind: Object.fromEntries([...new Set(curriculum.map(row => row.kind))].sort().map(kind => [kind,curriculum.filter(row => row.kind === kind).length])),
    projected_programs: programs.length, general_code_proposal_turns: nativeProposals.length, projection_rejected: projectionRejected.length,
    outputs: ['synthetic-code-tasks.jsonl','curriculum.jsonl','general-code.jsonl','code-proposals.jsonl','teacher-programs.jsonl','projection-rejected.jsonl'] };
  manifest.output_sha256=await outputHashes(outPath,manifest.outputs);
  await atomicWrite(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
  process.stdout.write(`${manifest.curriculum_tasks} curriculum tasks -> ${outPath}\n`);
}
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) main().catch(error => { process.stderr.write(`${error}\n`); process.exitCode = 1; });
