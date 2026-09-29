/** Bounded live smoke. This records failures; it is not the complete matched-search acceptance experiment. */
import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve, join } from 'node:path';
import { loadEvaluationSuite, evaluate, evaluationSummary } from '@natlang/node/evaluation';
import { loadModelConfiguration, createResolvedModelSession, executorIdentityForChoice } from '@natlang/node/model';
import { fingerprint } from '@natlang/node/adaptation';

const root = resolve(import.meta.dirname);
const out = resolve(process.argv[2] ?? join(root, 'evidence', 'live-2026-09-29.json'));
const configuration = loadModelConfiguration('default');
const model = createResolvedModelSession(configuration.choice, process.env, process.stderr);
const report: Record<string, unknown> = { schema: 'natlang.live-adaptation-smoke/v1', started: new Date().toISOString(),
  executor: executorIdentityForChoice(configuration.choice), status: 'running', suites: [],
  limitations: ['This bounded baseline smoke is not the complete matched GEPA/reflection/guidance/replicate acceptance experiment.'] };
const suites: unknown[] = [];
const save = () => { mkdirSync(resolve(out, '..'), { recursive: true }); writeFileSync(out, JSON.stringify({ ...report, suites }, null, 2) + '\n'); };
try {
  for (const name of ['triage', 'moderation', 'stateful']) {
    process.stderr.write(`Live baseline: ${name}\n`);
    const suite = await loadEvaluationSuite(join(root, name, 'suite.ts'));
    try {
      const batch = await evaluate(suite, { split: 'validation', driver: (request, signal) => model.turn(request, signal), seed: 929 });
      suites.push({ id: suite.suite.id, programHash: suite.program.buildHash, suiteHash: suite.suiteHash,
        summary: evaluationSummary(batch, suite.cases), evidenceDigest: batch.digest,
        cases: batch.results.map(result => ({ id: result.caseId, status: result.status, quality: result.quality, gates: result.gates,
          outcome: result.outcome, usage: result.usage, latencyMs: result.latencyMs, traceDigest: fingerprint(result.traces) })) });
    } catch (error) { suites.push({ id: suite.suite.id, status: 'infrastructure-error', error: String(error) }); }
    save();
  }
  report.status = 'completed-baseline-smoke'; report.finished = new Date().toISOString(); save();
} finally { await model.close(); }
console.log(out);
