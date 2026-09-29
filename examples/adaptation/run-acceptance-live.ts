/** Matched bounded searches and repeated held-out evaluation; all failures are retained. */
import { mkdirSync, writeFileSync, readFileSync, existsSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { loadEvaluationSuite, prepareEvaluationSuite, evaluate, evaluationSummary, replicateUncertainty, UsageGateway } from '@natlang/node/evaluation';
import { optimize } from '@natlang/node/optimize';
import { loadAdaptation, bindAdaptation } from '@natlang/node/adaptation';
import { loadModelConfiguration, createResolvedModelSession, executorIdentityForChoice } from '@natlang/node/model';
const root = resolve(import.meta.dirname), evidence = join(root, 'evidence');
const requested = process.argv.find(value => value.startsWith('--suites='))?.slice(9).split(',') ?? ['triage', 'moderation', 'stateful'];
if (requested.some(name => !['triage', 'moderation', 'stateful'].includes(name))) throw new Error('unknown acceptance suite');
const outputName = process.argv.find(value => value.startsWith('--output='))?.slice(9) ?? 'acceptance-live-2026-09-29.json';
if (!/^[A-Za-z0-9_.-]+\.json$/.test(outputName)) throw new Error('output must be a JSON filename');
const runTag = outputName.replace(/\.json$/, '');
const choice = loadModelConfiguration('default').choice, identity = executorIdentityForChoice(choice);
const executor = createResolvedModelSession(choice, process.env, process.stderr);
const reflection = createResolvedModelSession(choice, process.env, process.stderr);
const allowance = { maxRollouts: 20, maxProposals: 1, maxModelCalls: 80, maxElapsedMs: 180000 };
const records: unknown[] = [];
const save = () => { mkdirSync(evidence, { recursive: true }); writeFileSync(join(evidence, outputName), JSON.stringify({
  schema: 'natlang.live-acceptance/v1', executor: identity, reflection: identity, allowance, replicates: 2,
  priorSearch: 'search-live-2026-09-29.json',
  priorAcceptance: outputName === 'acceptance-live-2026-09-29.json' ? null : 'acceptance-live-2026-09-29.json',
  limitations: ['Small hand-authored fixtures and one proposal per strategy; this measures lifecycle and instability, not benchmark accuracy.',
    'Executor and reflection are the same explicitly configured local base model.', 'Unavailable usage/pricing remains unknown.'], records,
}, null, 2) + '\n'); };
try {
  for (const name of requested) {
    const original = await loadEvaluationSuite(join(root, name, 'suite.ts'));
    const modes = name === 'moderation' ? ['lambda-only', 'guidance-only', 'joint'] as const : [original.suite.components];
    for (const mode of modes) {
      const prepared = await prepareEvaluationSuite({ ...original.suite, components: mode }, original.moduleURL);
      for (const strategy of ['reflection', 'gepa'] as const) {
        const directory = join(root, '.natlang', runTag, name, String(mode), strategy);
        process.stderr.write(`Acceptance: ${name}/${mode}/${strategy}\n`);
        if (!existsSync(join(directory, 'report.json'))) {
          try {
            await optimize(prepared, { strategy, seed: 929, out: directory, budget: allowance, maxRepairs: 0,
              executor: (request, signal) => executor.turn(request, signal), reflectionIdentity: identity,
              reflection: (request, signal) => reflection.turn({ ...request, max_tokens: Math.min(request.max_tokens ?? 512, 512) }, signal) });
          } catch (error) { records.push({ suite: name, mode, strategy, phase: 'search', status: 'failed', error: String(error) }); save(); }
        }
        const report = existsSync(join(directory, 'report.json')) ? JSON.parse(readFileSync(join(directory, 'report.json'), 'utf8')) : null;
        const failure = existsSync(join(directory, 'failure-report.json')) ? JSON.parse(readFileSync(join(directory, 'failure-report.json'), 'utf8')) : null;
        const artifactPath = join(directory, 'selected.json');
        let artifact = existsSync(artifactPath) ? loadAdaptation(artifactPath) : null;
        let candidate;
        try { candidate = artifact ? bindAdaptation(artifact, prepared.program, identity).candidate : undefined; }
        catch (error) { records.push({ suite: name, mode, strategy, phase: 'binding', status: 'failed', error: String(error) }); artifact = null; save(); }
        if (artifact) writeFileSync(join(evidence, `${runTag}-${name}-${mode}-${strategy}.json`), JSON.stringify(artifact, null, 2) + '\n');
        records.push({ suite: name, mode, strategy, phase: 'search-report', status: artifact ? 'selected' : 'no-promotable-artifact',
          artifact: artifact?.digest ?? null, failure, report: report ? { stopReason: report.stopReason, improvement: report.improvement,
            summaries: report.summaries, uncovered: report.uncovered, ledger: report.ledger, history: report.history } : null }); save();
        // The finalist is already frozen. These fresh validation replicates cannot
        // feed reflection or choose a different candidate.
        for (const selection of ['baseline', 'selected'] as const) {
          if (selection === 'selected' && !artifact) continue;
          const batches = [];
          for (let replicate = 0; replicate < 2; replicate++) {
            const gateway = new UsageGateway(prepared.suite.budget);
            try {
              const batch = await evaluate(prepared, { split: 'validation', replicate, seed: 20260929,
                candidate: selection === 'selected' ? candidate : undefined, gateway, driver: (request, signal) => executor.turn(request, signal) });
              batches.push(batch);
              records.push({ suite: name, mode, strategy, phase: 'replicate', selection, replicate, status: 'scored',
                summary: evaluationSummary(batch, prepared.cases), digest: batch.digest, ledger: gateway.snapshot(),
                cases: batch.results.map(result => ({ id: result.caseId, quality: result.quality, gates: result.gates,
                  outcome: result.outcome, metrics: result.metrics, observation: result.observation, coverage: result.coverage, usage: result.usage })) });
            } catch (error) { records.push({ suite: name, mode, strategy, phase: 'replicate', selection, replicate, status: 'failed', error: String(error), ledger: gateway.snapshot() }); }
            save();
          }
          records.push({ suite: name, mode, strategy, phase: 'uncertainty', selection, ...replicateUncertainty(batches) }); save();
        }
      }
    }
  }
} finally { await reflection.close(); await executor.close(); }
console.log(join(evidence, outputName));
