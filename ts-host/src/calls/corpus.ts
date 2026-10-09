/**
 * The specializer's results as a training corpus (`natlang compilations export-corpus`): each compiled case with the
 * evidence that it does what the agent did (the calls it was checked on, with the judge's verdicts), each decline with
 * its reason and sample calls (where natural language was judged essential), and the findings. The corpus is evidence
 * about which parts of natural-language functions are crisp; whether it is used for training is decided separately,
 * in the corpus registry (training/neuralese_corpora.json).
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { caseServices, caseSources } from './compilations.js';
import type { CallStore } from './store.js';
import type { CallRecord } from './types.js';

export type CorpusSummary = { cases: number; declines: number; findings: number; calls: number; files: string[] };

const README = `# Specialization corpus

Exported by \`natlang compilations export-corpus\` from a machine's call store (ts-host/src/calls/corpus.ts).

- cases.jsonl: one compiled case per line: the function (name, source, revision key, instructions, signature), the case's
  TypeScript (guard and body), its tier and counts, and the recorded calls it was checked on. Each call has its inputs,
  the agent's result and service calls, the role of the check (training, held-out, shadow, audit, served) and the
  judge's verdict (equal, equivalent, better, worse).
- declines.jsonl: one decline per line: the function, the reason (semantic, unstable, effects, no-clusters, budget,
  not-worth-it), why, and sample calls with inputs, result and what they did.
- findings.jsonl: what compiling found about the program or its executor.

Calls are from real runs on this machine; inputs may contain conversation text. Copies and derived subsets are not
independent examples. Training admission is recorded in the corpus registry, not here.
`;

function callEvidence(store: CallStore, record: CallRecord): Record<string, unknown> {
  return { call: record.call_id, executor: record.executor.kind, model: record.executor.model_id,
    inputs: Object.fromEntries(Object.entries({ ...record.captures, ...record.inputs }).map(([name, ref]) => [name, store.value(ref) ?? ref])),
    result: store.value(record.output) ?? record.output, outcome: record.outcome,
    effects: record.effects.map(effect => ({ call: `${effect.service}.${effect.method}`, args: store.value(effect.args),
      ...(effect.error ? { error: effect.error } : { result: store.value(effect.result) }) })),
    evals: record.approach.evals, cost: record.cost };
}

/** Write the corpus files under `out`. `perCase` bounds the calls kept per case, `perDecline` per decline. */
export function exportSpecializationCorpus(store: CallStore, out: string, options: { definition?: string; perCase?: number; perDecline?: number } = {}): CorpusSummary {
  mkdirSync(out, { recursive: true });
  let calls = 0;
  const cases: string[] = [];
  for (const row of store.compilations({ definition: options.definition, status: 'current', limit: 10_000 })) {
    const compilation = store.compilation(row.id);
    if (!compilation?.files['cases.ts']) continue;
    const text = compilation.files['cases.ts'];
    const sources = caseSources(text), services = caseServices(text);
    const sample = compilation.cases.length ? store.call(store.caseCalls(compilation.cases[0]!.hash, undefined, 1)[0]?.call_id ?? '') : undefined;
    const definition = sample?.definition;
    for (const item of compilation.cases) {
      const links = store.caseCalls(item.hash, undefined, options.perCase ?? 40);
      const checked = links.map(link => { const record = store.call(link.call_id); return record ? { role: link.role, verdict: link.verdict, ...callEvidence(store, record) } : undefined; })
        .filter(Boolean);
      calls += checked.length;
      cases.push(JSON.stringify({ schema: 'natlang.specialization-case/1', compilation: row.id, created: row.created_at,
        definition: { name: row.definition_name, id: row.definition_id, source: row.definition_source, key: row.definition_key,
          instructions: definition ? store.value(definition.instructions) ?? null : null, params: definition?.params ?? null,
          returns: definition?.returns ?? null, types: definition?.types ?? null },
        case: { hash: item.hash, position: item.position, source: sources[item.position] ?? null, services: services[item.position] ?? [],
          tier: item.tier, served: item.served, handed_off: item.handed_off, compared: item.compared, worse: item.worse, better: item.better,
          live_compared: item.live_compared ?? 0, live_worse: item.live_worse ?? 0 },
        file: text, calls: checked }));
    }
  }
  const declines: string[] = [];
  for (const decline of store.declines(10_000)) {
    const sample = store.calls({ key: decline.definition_key, outcome: 'done', executor: 'agent', limit: options.perDecline ?? 8 });
    const records = sample.map(summary => store.call(summary.call_id)).filter((record): record is CallRecord => !!record);
    const definition = records[0]?.definition;
    if (options.definition && ![decline.definition_key, decline.definition_id, definition?.name, definition?.source].includes(options.definition)) continue;
    calls += records.length;
    declines.push(JSON.stringify({ schema: 'natlang.specialization-decline/1', created: decline.created_at, reason: decline.reason, why: decline.why,
      calls_at_decline: decline.calls_at_decline,
      definition: { name: definition?.name ?? null, id: decline.definition_id, source: definition?.source ?? null, key: decline.definition_key,
        instructions: definition ? store.value(definition.instructions) ?? null : null, params: definition?.params ?? null, returns: definition?.returns ?? null },
      calls: records.map(record => callEvidence(store, record)) }));
  }
  const findings = store.findings({ definition: options.definition, all: true, limit: 10_000 }).map(row => JSON.stringify({ schema: 'natlang.specialization-finding/1', ...row }));
  const write = (name: string, lines: string[]) => writeFileSync(join(out, name), lines.length ? lines.join('\n') + '\n' : '');
  write('cases.jsonl', cases); write('declines.jsonl', declines); write('findings.jsonl', findings);
  writeFileSync(join(out, 'README.md'), README);
  return { cases: cases.length, declines: declines.length, findings: findings.length, calls, files: ['README.md', 'cases.jsonl', 'declines.jsonl', 'findings.jsonl'] };
}
