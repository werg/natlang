#!/usr/bin/env node
/** Replay a versioned WorkflowEvals typed-return IR set through the pinned native runtime. */
import { createReadStream, readFileSync } from 'node:fs';
import { StringDecoder } from 'node:string_decoder';
import { createHash } from 'node:crypto';
import { open, writeFile } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { pathToFileURL } from 'node:url';

const [runtimeArg, inputArg, outputArg, reportArg] = process.argv.slice(2);
if (!runtimeArg || !inputArg || !outputArg || !reportArg)
  throw Error('usage: prove_workflow_typed_return_ir.mjs PINNED_RUNTIME INPUT.ir.jsonl RESULTS.jsonl REPORT.json');
const runtime = resolve(runtimeArg), input = resolve(inputArg), output = resolve(outputArg), reportPath = resolve(reportArg);
const runtimeManifestSha256 = createHash('sha256').update(readFileSync(join(runtime, 'frozen-runtime.json'))).digest('hex');
const load = relative => import(pathToFileURL(join(runtime, relative)).href);
const [curriculum, collector, prompt, conversion, materializer] = await Promise.all([
  load('dist/teacher/curriculum.js'), load('dist/teacher/collector.js'), load('dist/native/prompt.js'),
  load('dist/teacher/source-conversion.js'), load('dist/teacher/native-materializer.js'),
]);
const out = await open(output, 'wx');
const inputHash = createHash('sha256'), outputHash = createHash('sha256');
const counts = { cases: 0, admitted: 0, denied: 0, materialized: 0, unlinked: 0, approved_decisions: 0, held_decisions: 0 };
const rejected = [];
async function* exactLines(path) {
  const decoder = new StringDecoder('utf8'); let pending = '';
  for await (const chunk of createReadStream(path)) {
    inputHash.update(chunk); pending += decoder.write(chunk);
    let newline;
    while ((newline = pending.indexOf('\n')) >= 0) {
      yield pending.slice(0, newline).replace(/\r$/, ''); pending = pending.slice(newline + 1);
    }
  }
  pending += decoder.end();
  if (pending.length) yield pending.replace(/\r$/, '');
}
const options = { modelId: 'native-typed-return-reference-v1', rootSeed: 909, systemPrompt: prompt.TOOLS_PROMPT,
  contextTokens: 16384, maxTurns: 60, toolSurfaceSha256: await collector.defaultToolSurfaceHash(), collectionRole: 'reference' };
try {
  for await (const line of exactLines(input)) {
    if (!line.trim()) continue;
    const ir = JSON.parse(line), driver = curriculum.referenceDriver(ir), trajectory = [];
    const wrapped = async request => {
      const response = await driver(request), turn = collector.trajectoryTurn(request, response);
      trajectory.push(turn); return response;
    };
    const expected = collector.expectedProvenance(ir, options), runId = collector.programRunId(counts.cases, expected);
    const run = await collector.executeProgram(ir, wrapped, { ...options, runId });
    const row = collector.programRow(ir, options.modelId, runId, expected, run, trajectory);
    const admission = curriculum.admitRow(row), conversionProblems = conversion.sourceConversionProblems(row);
    const native = materializer.materializeNativeRows([row], { failedRuns: true });
    const approved = native.turns.filter(turn => turn.training_admission?.approved).length;
    const result = { id: ir.id, source_id: ir.external_source?.source_id ?? ir.source_ids?.[0] ?? null,
      program_sha256: createHash('sha256').update(JSON.stringify(ir)).digest('hex'),
      reference_status: run.outcome?.status, admission: admission.admitted ? 'admitted' : 'denied',
      admission_reasons: admission.reasons, source_conversion_problems: conversionProblems,
      materialized_turns: native.turns.length, approved_decisions: approved, held_decisions: native.turns.length - approved,
      unlinked: native.unlinked, outcome: run.outcome };
    const raw = `${JSON.stringify(result)}\n`; await out.writeFile(raw); outputHash.update(raw);
    counts.cases++; counts[admission.admitted ? 'admitted' : 'denied']++;
    counts.materialized += native.turns.length; counts.unlinked += native.unlinked.length;
    counts.approved_decisions += approved; counts.held_decisions += native.turns.length - approved;
    if (!admission.admitted || conversionProblems.length || native.unlinked.length || native.turns.length === 0 || approved === 0)
      rejected.push({ id: ir.id, admission: admission.reasons, conversionProblems, unlinked: native.unlinked, turns: native.turns.length, approved });
  }
  await out.sync(); await out.close();
  const report = { version: 'natlang.workflow_typed_return_native_proof/1', runtime, input, output,
    input_sha256: inputHash.digest('hex'), output_sha256: outputHash.digest('hex'),
    runtime_manifest_sha256: runtimeManifestSha256,
    model_calls: 0, options, counts, rejected };
  await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`, { flag: 'wx' });
  process.stdout.write(`${JSON.stringify({ ...report, rejected: rejected.length })}\n`);
} catch (error) { await out.close(); throw error; }
