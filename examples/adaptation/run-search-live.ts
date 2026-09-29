/** Small matched search smoke; larger replicated holdouts are a separate release gate. */
import { mkdirSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { loadEvaluationSuite } from '@natlang/node/evaluation';
import { optimize } from '@natlang/node/optimize';
import { loadModelConfiguration, createResolvedModelSession, executorIdentityForChoice } from '@natlang/node/model';
const root = resolve(import.meta.dirname), out = join(root, 'evidence', 'search-live-2026-09-29.json');
const choice = loadModelConfiguration('default').choice;
const executor = createResolvedModelSession(choice, process.env, process.stderr);
const reflection = createResolvedModelSession(choice, process.env, process.stderr);
const records: unknown[] = [];
const save = () => { mkdirSync(join(root, 'evidence'), { recursive: true }); writeFileSync(out, JSON.stringify({
  schema: 'natlang.live-search-smoke/v1', executor: executorIdentityForChoice(choice), reflection: executorIdentityForChoice(choice),
  allowance: { maxRollouts: 20, maxProposals: 1, maxModelCalls: 80, maxElapsedMs: 180000 },
  limitations: ['One proposal per strategy is a bounded smoke, not a replicated final acceptance experiment.', 'Reflection replies are bounded to 512 output tokens.'], records,
}, null, 2) + '\n'); };
try {
  for (const name of ['triage', 'moderation', 'stateful']) {
    const suite = await loadEvaluationSuite(join(root, name, 'suite.ts'));
    for (const strategy of ['reflection', 'gepa'] as const) {
      process.stderr.write(`Live search: ${name}/${strategy}\n`);
      try {
        const result = await optimize(suite, { strategy, seed: 929, out: join(root, '.natlang', 'live-search', name, strategy),
          executor: (request, signal) => executor.turn(request, signal), reflection: (request, signal) => reflection.turn({ ...request, max_tokens: Math.min(request.max_tokens ?? 512, 512) }, signal),
          budget: { maxRollouts: 20, maxProposals: 1, maxModelCalls: 80, maxElapsedMs: 180000 }, maxRepairs: 0,
          progress: event => process.stderr.write(JSON.stringify(event) + '\n') });
        records.push({ suite: name, strategy, status: 'completed', artifact: result.artifact.digest,
          summary: { stopReason: result.state.stopReason, improvement: result.report.improvement, ledger: result.state.ledger, uncovered: result.report.uncovered,
            baselineQuality: result.report.baseline, validation: result.report.summaries, history: result.state.history } });
      } catch (error) { records.push({ suite: name, strategy, status: 'failed', error: String(error) }); }
      save();
    }
  }
} finally { await reflection.close(); await executor.close(); }
console.log(out);
