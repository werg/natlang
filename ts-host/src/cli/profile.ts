/**
 * `natlang profile [DIR] [--store PATH] [--policy crisp|nl|shadow] [--profile NAME] [--json]` (WS2 R5):
 * aggregates per-function metrics from the CallStore (volume, p50 latency, avg tokens, compiled share,
 * decision confidence p10/p50, escalation rate from `decision_escalate` events, failure classes) and
 * invokes the pluggable `nextOptimization` policy to recommend the next optimization step.
 */
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';
import type { CallStore, CallSummary } from '../calls/store.js';
import { machineStoreRoot, openMachineStore } from '../machine-store.js';
import { decisionDomain } from '../native/decision.js';
import { parseAliases, parseTypeCached, TypeEnv } from '../native/types.js';
import { nextOptimization, type FunctionProfile, type OptimizationRecommendation } from '../optimization/policies.js';
import type { PluggableSetting } from '../runtime/pluggable.js';
import { usageError } from './args.js';

export type ProfileEntry = FunctionProfile & {
  definitionId: string;
  failureClasses: Record<string, number>;
  escalations: number;
  compiledCalls: number;
  recommendation: OptimizationRecommendation;
};

export type ProfileReport = {
  program: string | null;
  store: string;
  totalCalls: number;
  functions: ProfileEntry[];
};

function percentile(sorted: readonly number[], p: number): number | null {
  if (!sorted.length) return null;
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil(p * sorted.length) - 1));
  return sorted[index]!;
}

function round(value: number, digits = 3): number {
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

function isFiniteReturnType(returns: string | undefined, types: Record<string, string> | undefined): boolean {
  if (!returns) return false;
  try {
    const env = new TypeEnv(parseAliases(types ?? {}));
    const domain = decisionDomain(parseTypeCached(returns), env);
    return domain !== null && domain.size >= 2;
  } catch {
    return false;
  }
}

/** Extract scored decision confidences from a call's trace events. */
function decisionConfidences(events: readonly Record<string, unknown>[]): number[] {
  const out: number[] = [];
  for (const event of events) {
    if (event.kind !== 'decision_readout') continue;
    if (typeof event.confidence === 'number' && Number.isFinite(event.confidence)) {
      out.push(event.confidence);
      continue;
    }
    if (event.phase === 'scored' && Array.isArray(event.probabilities) && typeof event.chosen === 'number') {
      const value = event.probabilities[event.chosen];
      if (typeof value === 'number' && Number.isFinite(value)) out.push(value);
    }
  }
  return out;
}

/** Aggregate per-function profiles and recommendations from `store`, optionally filtered to `options.dir`. */
export async function profileStore(
  store: CallStore,
  options: {
    dir?: string;
    limit?: number;
    policy?: PluggableSetting;
    nl?: (profile: FunctionProfile) => OptimizationRecommendation | Promise<OptimizationRecommendation>;
  } = {},
): Promise<ProfileReport> {
  const programDir = options.dir ? resolve(options.dir) : undefined;
  const rows = store.calls({ program: programDir, limit: options.limit ?? 10_000 });

  // Index escalation spans (e.g. `std:escalate`) whose events carry `decision_escalate` for child calls.
  const escalatedParents = new Set<string>();
  for (const row of rows) {
    if (row.definition_id === 'std:escalate' || row.definition_name === 'std escalate') {
      const events = store.events(row.call_id) ?? [];
      if (events.some(event => event.kind === 'decision_escalate')) escalatedParents.add(row.call_id);
    }
  }

  type Bucket = {
    name: string;
    definitionId: string;
    source: string | null;
    programDir?: string;
    calls: CallSummary[];
    latencies: number[];
    totalTokens: number;
    compiledCalls: number;
    confidences: number[];
    escalations: number;
    countedEscalationParents: Set<string>;
    failures: number;
    failureClasses: Record<string, number>;
    hasDecisionReadout: boolean;
    finiteReturn: boolean;
  };

  const buckets = new Map<string, Bucket>();
  let totalCalls = 0;

  for (const row of rows) {
    if (row.outcome === 'running') continue;
    if (row.definition_id.startsWith('std:') || row.definition_id.startsWith('pluggable:')) continue;
    const record = store.call(row.call_id);
    if (record?.definition.site === 'internal') continue;
    totalCalls++;

    const key = row.definition_source || row.definition_name || row.definition_id;
    let bucket = buckets.get(key);
    if (!bucket) {
      bucket = {
        name: row.definition_name || row.definition_id,
        definitionId: row.definition_id,
        source: row.definition_source,
        programDir: programDir ?? row.program_root ?? undefined,
        calls: [],
        latencies: [],
        totalTokens: 0,
        compiledCalls: 0,
        confidences: [],
        escalations: 0,
        countedEscalationParents: new Set(),
        failures: 0,
        failureClasses: {},
        hasDecisionReadout: false,
        finiteReturn: false,
      };
      buckets.set(key, bucket);
    }
    bucket.calls.push(row);
    bucket.latencies.push(Number(row.wall_ms) || 0);
    bucket.totalTokens += (Number(row.tokens_in) || 0) + (Number(row.tokens_out) || 0);
    if (row.executor === 'crisp') bucket.compiledCalls++;
    if (row.outcome !== 'done') {
      bucket.failures++;
      const cls = row.outcome || 'failed';
      bucket.failureClasses[cls] = (bucket.failureClasses[cls] ?? 0) + 1;
    }

    if (record) {
      if (record.definition.readout === 'decision' || record.definition.readout === 'instant') {
        bucket.hasDecisionReadout = true;
      }
      if (!bucket.finiteReturn && isFiniteReturnType(record.definition.returns, record.definition.types)) {
        bucket.finiteReturn = true;
      }
    }

    const events = store.events(row.call_id) ?? [];
    const confs = decisionConfidences(events);
    if (confs.length) {
      bucket.hasDecisionReadout = true;
      bucket.confidences.push(...confs);
    }
    if (events.some(event => event.kind === 'decision_escalate')) {
      bucket.escalations++;
    } else if (row.parent_call_id && escalatedParents.has(row.parent_call_id) && !bucket.countedEscalationParents.has(row.parent_call_id)) {
      bucket.countedEscalationParents.add(row.parent_call_id);
      bucket.escalations++;
    }
  }

  const functions: ProfileEntry[] = [];
  for (const bucket of buckets.values()) {
    const count = bucket.calls.length;
    const sortedLatencies = bucket.latencies.slice().sort((a, b) => a - b);
    const sortedConfs = bucket.confidences.slice().sort((a, b) => a - b);
    const p50Ms = percentile(sortedLatencies, 0.5) ?? 0;
    const avgTokens = count ? round(bucket.totalTokens / count, 1) : 0;
    const compiledShare = count ? round(bucket.compiledCalls / count, 4) : 0;
    const decisionP10 = sortedConfs.length ? round(percentile(sortedConfs, 0.1)!, 4) : null;
    const decisionP50 = sortedConfs.length ? round(percentile(sortedConfs, 0.5)!, 4) : null;
    const escalationRate = count ? round(bucket.escalations / count, 4) : 0;
    const failureRate = count ? round(bucket.failures / count, 4) : 0;

    const profile: FunctionProfile = {
      name: bucket.name,
      calls: count,
      p50Ms,
      avgTokens,
      compiledShare,
      decisionP10,
      decisionP50,
      escalationRate,
      failureRate,
      hasDecisionReadout: bucket.hasDecisionReadout,
      finiteReturn: bucket.finiteReturn,
      ...(bucket.programDir ? { programDir: bucket.programDir } : {}),
      ...(bucket.source !== null ? { source: bucket.source } : {}),
    };
    const recommendation = await nextOptimization(profile, { policy: options.policy, nl: options.nl });
    functions.push({
      ...profile,
      definitionId: bucket.definitionId,
      failureClasses: bucket.failureClasses,
      escalations: bucket.escalations,
      compiledCalls: bucket.compiledCalls,
      recommendation,
    });
  }

  functions.sort((a, b) => b.calls - a.calls || a.name.localeCompare(b.name));
  return { program: programDir ?? null, store: store.root, totalCalls, functions };
}

function formatTable(rows: Record<string, string>[], columns: string[]): string {
  if (!rows.length) return '(no recorded calls)';
  const cells = [columns, ...rows.map(row => columns.map(col => row[col] ?? ''))];
  const widths = columns.map((_, index) => Math.min(60, Math.max(...cells.map(row => row[index]!.length))));
  return cells.map(row => row.map((cell, index) => cell.slice(0, 60).padEnd(widths[index]!)).join('  ').trimEnd()).join('\n');
}

/** Render a human-readable profile table and recommendations. */
export function renderProfileReport(report: ProfileReport): string {
  if (!report.functions.length) {
    return report.program ? `no recorded calls for ${report.program}` : 'no recorded calls in store';
  }
  const pct = (value: number) => `${(value * 100).toFixed(0)}%`;
  const rows = report.functions.map(fn => ({
    function: fn.source ?? fn.name,
    calls: String(fn.calls),
    p50_ms: String(Math.round(fn.p50Ms)),
    avg_tok: String(Math.round(fn.avgTokens)),
    compiled: pct(fn.compiledShare),
    conf_p10: fn.decisionP10 === null ? '-' : fn.decisionP10.toFixed(2),
    conf_p50: fn.decisionP50 === null ? '-' : fn.decisionP50.toFixed(2),
    escalate: pct(fn.escalationRate),
    fail: pct(fn.failureRate),
    next: fn.recommendation.action,
  }));
  const lines = [
    formatTable(rows, ['function', 'calls', 'p50_ms', 'avg_tok', 'compiled', 'conf_p10', 'conf_p50', 'escalate', 'fail', 'next']),
  ];
  const actionable = report.functions.filter(fn => fn.recommendation.action !== 'none');
  if (actionable.length) {
    lines.push('', 'Recommendations:');
    for (const fn of actionable) {
      lines.push(`  ${fn.source ?? fn.name}: ${fn.recommendation.command ?? fn.recommendation.action}`);
      lines.push(`    ${fn.recommendation.reason}`);
    }
  }
  return lines.join('\n');
}

/** `natlang profile [DIR] [--store PATH] [--policy crisp|nl|shadow] [--profile NAME] [--json]` */
export async function profileCommand(args: string[]): Promise<number> {
  let dir: string | undefined;
  let storePath: string | undefined;
  let policy: PluggableSetting;
  let json = false;
  let profile: string | undefined;
  for (let index = 0; index < args.length; index++) {
    const item = args[index]!;
    if (item === '--json') { json = true; continue; }
    if (item === '--profile' || item.startsWith('--profile=')) {
      const value = item.includes('=') ? item.slice(item.indexOf('=') + 1) : args[++index];
      if (!value || value.startsWith('--')) throw usageError('--profile needs a model profile name');
      profile = value;
      continue;
    }
    if (item === '--store' || item.startsWith('--store=')) {
      const value = item.includes('=') ? item.slice(item.indexOf('=') + 1) : args[++index];
      if (!value || (!item.includes('=') && value.startsWith('--'))) throw usageError('--store needs a directory path');
      storePath = value;
      continue;
    }
    if (item === '--policy' || item.startsWith('--policy=')) {
      const value = item.includes('=') ? item.slice(item.indexOf('=') + 1) : args[++index];
      if (!value || !['crisp', 'nl', 'shadow'].includes(value)) throw usageError('--policy must be crisp, nl, or shadow');
      policy = value as PluggableSetting;
      continue;
    }
    if (item.startsWith('-')) throw usageError(`unknown option ${item} for natlang profile; usage: natlang profile [DIR] [--store PATH] [--policy crisp|nl|shadow] [--profile NAME] [--json]`);
    if (dir !== undefined) throw usageError(`unexpected argument ${item}; usage: natlang profile [DIR] [--store PATH] [--json]`);
    dir = item;
  }
  if (dir && !existsSync(resolve(dir))) throw usageError(`${dir}: no such directory`);
  const root = storePath ? resolve(storePath) : machineStoreRoot();
  if (!root) throw new Error('the call store is off on this machine (NATLANG_CALL_STORE=off); pass --store PATH to read one');
  const store = openMachineStore(root);
  try {
    // The natural-language side of nextOptimization is a natlang call: it runs on the configured model.
    let report: ProfileReport;
    if (policy === 'nl' || policy === 'shadow') {
      const { natlangRuntime } = await import('../configured-runtime.js');
      const configured = await natlangRuntime({ ...(profile ? { profile } : {}), escalate: false });
      try { report = await configured.runtime.run(() => profileStore(store, { dir, policy })); }
      finally { await configured.close(); }
    } else report = await profileStore(store, { dir, policy });
    process.stdout.write(json ? `${JSON.stringify(report, null, 2)}\n` : `${renderProfileReport(report)}\n`);
    return 0;
  } finally {
    store.close();
  }
}
