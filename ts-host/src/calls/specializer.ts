/**
 * The crisp half of the specializer (§5, §7.5, §7.6): examples from the store, the evidence folder, verification of a
 * cases file against recorded calls, and saving an accepted compilation. The judgments (grouping, conditions, whether
 * to compile, the TypeScript itself) are the natural-language reducer's (applications/specializer). Node only.
 */
import ts from 'typescript';
import type { CallableDefinition } from '../runtime/kernel.js';
import type { NatlangRuntime } from '../runtime/runtime.js';
import { caseHashes, caseSources, loadCases } from './compilations.js';
import { recordedBehavior, type Behavior } from './judge.js';
import { approaches, behaviorLabel, induceRules, splitOf, type Approach, type Example, type Rule } from './mining.js';
import { definitionFor, replayCaseOn, signatureOf, verdictFor } from './offline.js';
import { recordedArguments } from './replay.js';
import type { CallStore } from './store.js';
import { isModelEvidence, type CallRecord, type CaseRole, type DeclineReason, type Verdict } from './types.js';

/** Everything the specializer knows about one definition revision. */
export type Study = { key: string; definition: CallableDefinition; loaded: boolean; records: Map<string, CallRecord>; examples: Example[];
  approaches: Approach[]; rules: Rule[]; unclassified: string[]; skipped: number };

/** Successful agent calls of a definition revision as examples, with approaches and induced conditions. */
export function study(store: CallStore, key: string, options: { limit?: number; minSupport?: number } = {}): Study | undefined {
  const summaries = store.agentCalls(key, options.limit ?? 2000).filter(call => call.outcome === 'done' && call.executor === 'agent');
  const records = new Map<string, CallRecord>();
  const examples: Example[] = [];
  let skipped = 0;
  for (const summary of summaries) {
    const record = store.call(summary.call_id);
    const args = record && recordedArguments(store, record);
    if (!record || !args) { skipped++; continue; }
    // A scripted test agent's calls are fixtures, not what a model does (§6.2: cases follow the evidence's executor).
    if (!isModelEvidence(record)) { skipped++; continue; }
    records.set(record.call_id, record);
    examples.push({ callId: record.call_id, args, features: record.features, evals: evalsOf(record), approach: '', split: splitOf(record.call_id),
      behavior: behaviorOf(store, record) });
  }
  if (!examples.length) return undefined;
  const found = approaches(examples);
  const byCall = new Map(found.flatMap(approach => approach.calls.map(id => [id, approach.id] as const)));
  for (const example of examples) example.approach = byCall.get(example.callId)!;
  const { rules, unclassified } = induceRules(examples, { minSupport: options.minSupport });
  const latest = records.get(examples.at(-1)!.callId)!;
  const { definition, loaded } = definitionFor(store, latest);
  return { key, definition, loaded, records, examples, approaches: found, rules, unclassified, skipped };
}

/** What a call did: its service calls and the functions it called, with those of its children, in order. */
function behaviorOf(store: CallStore, record: CallRecord, depth = 0): string {
  const steps: string[] = record.effects.filter(effect => !effect.error).sort((a, b) => a.order - b.order).map(effect => `${effect.service}.${effect.method}`);
  if (depth < 4) for (const child of store.children(record.call_id)) steps.push(child.definition_name);
  return behaviorLabel(steps);
}

/** The successful eval programs of a recorded agent call, in order. */
function evalsOf(record: CallRecord): string[] {
  return record.approach.evals.filter(Boolean);
}

/** A crisp reason to decline before any model is asked (§5.3), or undefined. */
export function crispDecline(subject: Study, minCalls: number): { reason: DeclineReason; why: string } | undefined {
  if (subject.examples.length < minCalls)
    return { reason: 'not-worth-it', why: `${subject.examples.length} recorded calls with exact inputs; at least ${minCalls} are needed` };
  const repeated = subject.approaches.filter(approach => approach.calls.length >= 3);
  if (!repeated.length && !subject.rules.length)
    return { reason: 'no-clusters', why: `no approach repeats in ${subject.examples.length} calls and no condition selects one` };
  return undefined;
}

// --- Evidence folder ------------------------------------------------------------------------------------------------

const json = (value: unknown) => JSON.stringify(value, null, 2);
const fence = (text: string, language = '') => `\`\`\`${language}\n${text}\n\`\`\``;

/** The evidence folder's files (§7.6). `report` is the previous verification round, `history` earlier outcomes. */
export function renderEvidence(store: CallStore, subject: Study, extra: { report?: string; history?: string; previousCases?: string } = {}): Record<string, string> {
  const files: Record<string, string> = { 'function.md': renderFunction(subject) };
  renderApproaches(store, subject, files);
  return renderRest(store, subject, files, extra);
}

/** function.md: the function's instructions, signature, what a case may call, and its volume. */
export function renderFunction(subject: Study): string {
  const { definition } = subject;
  const callees = Object.keys(definition.codebase);
  const cost = [...subject.records.values()].reduce((sum, record) => ({ tokens: sum.tokens + record.cost.tokens_in + record.cost.tokens_out,
    wall: sum.wall + record.cost.wall_ms }), { tokens: 0, wall: 0 });
  const count = subject.examples.length;
  return [`# ${definition.name}`, '', `Signature: \`${signatureOf(definition)}\``, '',
    `Source: ${definition.source ?? '(inline)'}${subject.loaded ? '' : ' (rebuilt from the record; its context is not available)'}`, '',
    '## Instructions', '', definition.body.trim(), '',
    ...(Object.keys(definition.types).length ? ['## Types', '', fence(Object.entries(definition.types).map(([name, text]) => `type ${name} = ${text};`).join('\n'), 'ts'), ''] : []),
    '## What a case may call', '',
    callees.length ? `Items of the function's context, imported by name (\`import helper from './helper'\`): ${callees.join(', ')}.` : 'No context items.',
    `Services, imported from 'natlang:services': ${[...new Set([...subject.records.values()].flatMap(record => record.effects.map(effect => effect.service)))].join(', ') || 'none recorded'}.`, '',
    '## Volume', '', `${count} successful agent calls with exact inputs (${subject.skipped} more could not be used), ` +
    `${Math.round(cost.tokens / Math.max(1, count))} tokens and ${Math.round(cost.wall / Math.max(1, count))} ms per call on average.`, '',
    `Split: ${subject.examples.filter(example => example.split === 'training').length} training, ` +
    `${subject.examples.filter(example => example.split === 'held-out').length} held out (used only to check your cases).`, ''].join('\n');
}

function renderApproaches(store: CallStore, subject: Study, files: Record<string, string>): void {
  // An approach taken once cannot make a case; it gets its line here but no folder, so the evidence fits the executor's
  // context (traces.call(id) shows such a call in full).
  const overview = ['# Approaches', '', 'Each approach is a group of calls whose code was the same after normalization. One line each.',
    'Approaches with one call have no folder: one call is not a pattern. traces.call(id) shows such a call in full.', '',
    '| approach | calls | what they did | a call | its result | code |', '| --- | --- | --- | --- | --- | --- |'];
  for (const approach of subject.approaches) {
    const first = subject.records.get(approach.calls[0]!);
    const example = subject.examples.find(item => item.callId === approach.calls[0]);
    const cell = (value: unknown, limit: number) => (JSON.stringify(value) ?? '').replace(/\|/g, '\\|').slice(0, limit);
    overview.push(`| ${approach.calls.length > 1 ? approach.id : `${approach.id} (${approach.calls[0]})`} | ${approach.calls.length} | ${example?.behavior ?? ''} | ${cell(example?.args, 80)} | ` +
      `${cell(first ? store.value(first.output) : null, 60)} | ${(approach.template[0] ?? '(no code)').replace(/\|/g, '\\|').slice(0, 100)} |`);
  }
  files['approaches/README.md'] = overview.join('\n') + '\n';
  for (const approach of subject.approaches) {
    if (approach.calls.length < 2) continue;
    const dir = `approaches/${approach.id}`;
    const calls = approach.calls.map(id => subject.records.get(id)!).filter(Boolean);
    files[`${dir}/approach.ts`] = approach.answerOnly ? '// The agent answered without running any code.\n' :
      [`// ${approach.calls.length} calls ran these evals (normalized: $in.<path> is that input, v0, v1 ... are locals,`,
        `// $h0, $h1 ... are literals that differed between calls; see stats.md).`, '',
        ...approach.template.map((program, index) => `// eval ${index + 1}\n${program}\n`)].join('\n');
    const outputs = new Map<string, number>();
    for (const record of calls) { const key = JSON.stringify(store.value(record.output)); outputs.set(key, (outputs.get(key) ?? 0) + 1); }
    const behaviorsHere = [...new Set(approach.calls.map(id => subject.examples.find(example => example.callId === id)?.behavior ?? 'none'))];
    files[`${dir}/stats.md`] = [`# ${approach.id}`, '', `${approach.calls.length} calls (${calls.filter(record => splitOf(record.call_id) === 'held-out').length} held out).`,
      `What they did: ${behaviorsHere.join('; ')}.`, '',
      'Most common results:', ...[...outputs].sort((a, b) => b[1] - a[1]).slice(0, 8).map(([value, n]) => `- ${n}× ${value.slice(0, 300)}`), '',
      ...(Object.values(approach.holes)[0]?.length ? ['Hole values per call:', ...Object.entries(approach.holes).slice(0, 20)
        .map(([id, values]) => `- ${id}: ${values.join(', ')}`), ''] : [])].join('\n');
    for (const record of calls.filter(record => splitOf(record.call_id) === 'training').slice(0, 5))
      files[`${dir}/examples/${record.call_id.replace(/[^\w.-]/g, '_')}.json`] = json(exampleOf(store, record));
  }
}

function renderRest(store: CallStore, subject: Study, files: Record<string, string>, extra: { report?: string; history?: string; previousCases?: string }): Record<string, string> {
  const behaviors = new Map<string, Example[]>();
  for (const example of subject.examples) behaviors.set(example.behavior ?? 'none', [...(behaviors.get(example.behavior ?? 'none') ?? []), example]);
  files['conditions.md'] = ['# Conditions', '',
    'What the calls did, whatever code they ran (their service calls and the functions they called):', '',
    ...[...behaviors].sort((a, b) => b[1].length - a[1].length).map(([behavior, list]) => `- ${behavior}: ${list.length} calls, ` +
      `approaches ${[...new Set(list.map(example => example.approach))].join(', ')}`), '',
    'Conditions found by search over the training calls: each selects one behavior with no exception among the training calls',
    'not covered by an earlier condition. They are evaluated in order, like the cases of a cases file. `held out` shows how',
    'the condition does on calls it was not chosen on. Use, change, merge or drop them; they are evidence, not the answer.', '',
    ...subject.rules.flatMap((rule, index) => [`## ${index + 1}. ${rule.label} (${rule.covers.length} training calls; held out: ` +
      `${rule.heldOut.correct}/${rule.heldOut.covered} correct)`, '', fence(rule.guard, 'ts'), '']),
    ...(subject.rules.length ? [] : ['No condition selects a behavior without exception.', '']),
    `${subject.unclassified.length} training calls are not covered by any condition (unclassified/).`, ''].join('\n');
  files['unclassified/README.md'] = `${subject.unclassified.length} training calls no condition covers. A few are listed here.\n`;
  for (const id of subject.unclassified.slice(0, 10)) {
    const record = subject.records.get(id);
    const example = subject.examples.find(item => item.callId === id);
    if (record) files[`unclassified/${id.replace(/[^\w.-]/g, '_')}.json`] = json({ approach: example?.approach, behavior: example?.behavior, ...exampleOf(store, record) });
  }
  files['history.md'] = extra.history ?? 'No earlier compilation or decline for this function.\n';
  if (extra.previousCases) files['previous-cases.ts'] = extra.previousCases;
  files['report.md'] = extra.report ?? 'No cases have been checked yet.\n';
  return files;
}

/** One recorded call as the evidence shows it. */
export function exampleOf(store: CallStore, record: CallRecord): Record<string, unknown> {
  return { call: record.call_id,
    inputs: Object.fromEntries(Object.entries({ ...record.captures, ...record.inputs }).map(([name, ref]) => [name, store.value(ref) ?? ref])),
    result: store.value(record.output),
    effects: record.effects.map(effect => ({ call: `${effect.service}.${effect.method}`, args: store.value(effect.args),
      ...(effect.error ? { error: effect.error } : { result: store.value(effect.result) }) })),
    evals: record.approach.evals, children: store.children(record.call_id).map(child => child.definition_name) };
}

/** History for the evidence folder: the current compilation's cases and numbers, hand-offs, declines. */
export function renderHistory(store: CallStore, key: string): { history: string; previousCases?: string } {
  const current = store.currentCompilation(key);
  const decline = store.declineFor(key);
  const lines: string[] = ['# History', ''];
  if (decline) lines.push(`Declined before (${decline.created_at}): ${decline.reason}: ${decline.why}`, '');
  if (current) {
    lines.push(`Current compilation ${current.id} (${current.created_at}); its cases.ts is in previous-cases.ts.`, '');
    for (const item of current.cases) {
      lines.push(`- case ${item.position + 1} (${item.hash}): ${item.tier}; served ${item.served}, handed back ${item.handed_off}, ` +
        `compared ${item.compared} (${item.worse} worse, ${item.better} better; ${item.live_compared ?? 0} live, ${item.live_worse ?? 0} worse), audited ${item.audited} (${item.audit_worse} worse)`);
      for (const link of store.caseCalls(item.hash, 'handed-off', 5)) {
        const record = store.call(link.call_id);
        if (record?.executor.case_error) lines.push(`  - handed back on ${link.call_id}: ${record.executor.case_error.slice(0, 300)}`);
      }
      for (const link of store.caseCalls(item.hash, undefined, 50).filter(link => link.verdict === 'worse' || link.verdict === 'diverged').slice(0, 5))
        lines.push(`  - ${link.role} on ${link.call_id}: ${link.verdict}`);
    }
    lines.push('');
  }
  if (!decline && !current) lines.push('No earlier compilation or decline for this function.');
  return { history: lines.join('\n') + '\n', ...(current?.files['cases.ts'] ? { previousCases: current.files['cases.ts'] } : {}) };
}

// --- Verification ---------------------------------------------------------------------------------------------------

export type CaseCheck = { position: number; hash: string; source: string; admitted: number; heldOutAdmitted: number;
  results: { callId: string; split: 'training' | 'held-out'; verdict: Verdict | 'error' | 'skipped'; detail?: string }[];
  guardErrors: number; accepted: boolean; reason: string };

/**
 * Check every case of a cases file on the recorded calls its guard admits (the first admitting case takes a call, as at
 * runtime). Equal behavior passes; a difference goes to the judge. A case is accepted when it admits at least one call
 * and no more than `bound` of its calls are worse, failed, or diverged.
 */
export async function verifyCases(runtime: NatlangRuntime, store: CallStore, subject: Study, text: string,
  options: { bound?: number; maxJudged?: number; signal?: AbortSignal } = {}): Promise<{ checks: CaseCheck[]; error?: string }> {
  let cases;
  try { cases = loadCases(`candidate-${subject.key.slice(0, 8)}`, text, subject.definition.codebase, subject.definition.types); }
  catch (error) { return { checks: [], error: error instanceof Error ? error.message : String(error) }; }
  const sources = caseSources(text);
  const checks: CaseCheck[] = cases.map((item, position) => ({ position, hash: item.hash, source: sources[position]!, admitted: 0, heldOutAdmitted: 0,
    results: [], guardErrors: 0, accepted: false, reason: '' }));
  const judged = new Map<number, number>();
  for (const example of subject.examples) {
    if (options.signal?.aborted) break;
    const at = cases.findIndex((item, position) => {
      try { return item.when(example.args) === true; } catch { checks[position]!.guardErrors++; return false; }
    });
    if (at < 0) continue;
    const check = checks[at]!;
    check.admitted++;
    if (example.split === 'held-out') check.heldOutAdmitted++;
    const record = subject.records.get(example.callId)!;
    const crisp = await replayCaseOn(runtime, store, cases[at]!, record);
    if ('skipped' in crisp) { check.results.push({ callId: example.callId, split: example.split, verdict: 'skipped', detail: crisp.skipped }); continue; }
    if (crisp.error !== undefined) { check.results.push({ callId: example.callId, split: example.split, verdict: 'error', detail: crisp.error }); continue; }
    const agent: Behavior = recordedBehavior(store, record);
    const count = judged.get(at) ?? 0;
    try {
      const { verdict, differences } = count < (options.maxJudged ?? 25) || example.split === 'held-out' ?
        await verdictFor(runtime, store, subject.definition, record, agent, crisp) :
        { verdict: 'diverged' as Verdict, differences: ['not judged: too many differing calls for this case'] };
      if (verdict !== 'equal') judged.set(at, count + 1);
      check.results.push({ callId: example.callId, split: example.split, verdict, ...(differences.length ? { detail: differences.join('; ') } : {}) });
    } catch (error) {
      check.results.push({ callId: example.callId, split: example.split, verdict: 'skipped', detail: `judge failed: ${error instanceof Error ? error.message : String(error)}` });
    }
  }
  const bound = options.bound ?? 0.05;
  for (const check of checks) {
    const bad = check.results.filter(result => ['worse', 'error', 'diverged'].includes(result.verdict)).length;
    const counted = check.results.filter(result => result.verdict !== 'skipped').length;
    check.accepted = counted > 0 && bad <= bound * counted && check.guardErrors === 0;
    check.reason = !check.admitted ? 'its guard admits none of the recorded calls' : !counted ? 'none of its calls could be replayed' :
      check.guardErrors ? `its guard threw on ${check.guardErrors} calls` : bad > bound * counted ? `${bad} of ${counted} calls worse, failed or diverged` :
      `${counted - bad} of ${counted} calls equal or better`;
  }
  return { checks };
}

/** The verification report written into the evidence folder for the next round (report.md). */
export function renderReport(store: CallStore, subject: Study, result: { checks: CaseCheck[]; error?: string }): string {
  if (result.error) return `# Report\n\nThe cases file does not load: ${result.error}\n\nFix it and keep the shape \`export const cases = [{ when: (args) => ..., run: async (args) => ... }]\`.\n`;
  const lines = ['# Report', '', 'Each case was run on the recorded calls its guard admits, against the recorded effects, and compared with what the agent did.', ''];
  for (const check of result.checks) {
    lines.push(`## Case ${check.position + 1}: ${check.accepted ? 'accepted' : 'not accepted'}: ${check.reason}`, '',
      `Admitted ${check.admitted} calls (${check.heldOutAdmitted} held out).`, '');
    const counts = new Map<string, number>();
    for (const result of check.results) counts.set(result.verdict, (counts.get(result.verdict) ?? 0) + 1);
    lines.push([...counts].map(([verdict, n]) => `${verdict}: ${n}`).join(', ') || 'No results.', '');
    for (const result of check.results.filter(item => !['equal', 'better', 'equivalent'].includes(item.verdict)).slice(0, 6)) {
      const record = subject.records.get(result.callId);
      lines.push(`- ${result.callId} (${result.split}): ${result.verdict}${result.detail ? `: ${result.detail}` : ''}`);
      if (record) lines.push(`  inputs: ${JSON.stringify(exampleOf(store, record).inputs).slice(0, 400)}`,
        `  agent result: ${JSON.stringify(store.value(record.output))?.slice(0, 300)}`);
    }
    lines.push('');
  }
  const admittedIds = new Set(result.checks.flatMap(check => check.results.map(item => item.callId)));
  lines.push(`${subject.examples.length - admittedIds.size} of ${subject.examples.length} calls are left to the agent.`, '');
  return lines.join('\n');
}

/** A cases file with only the cases at `keep` (by position), the rest of the file unchanged. */
export function keepCases(text: string, keep: readonly number[]): string {
  const file = ts.createSourceFile('cases.ts', text, ts.ScriptTarget.ES2022, true, ts.ScriptKind.TS);
  let array: ts.ArrayLiteralExpression | undefined;
  const visit = (node: ts.Node): void => {
    if (array) return;
    if (ts.isVariableDeclaration(node) && ts.isIdentifier(node.name) && node.name.text === 'cases' && node.initializer) {
      let initializer: ts.Expression = node.initializer;
      while (ts.isAsExpression(initializer) || ts.isSatisfiesExpression(initializer) || ts.isParenthesizedExpression(initializer)) initializer = initializer.expression;
      if (ts.isArrayLiteralExpression(initializer)) array = initializer;
    }
    ts.forEachChild(node, visit);
  };
  visit(file);
  if (!array) return text;
  const kept = array.elements.filter((_element, index) => keep.includes(index)).map(element => `  ${element.getText(file)},`);
  return `${text.slice(0, array.getStart(file))}[\n${kept.join('\n')}\n]${text.slice(array.getEnd())}`;
}

/**
 * Save the accepted cases of a verified cases file as the definition's current compilation, with links from each case
 * to the calls it covers and its held-out verdicts counted toward promotion (§6.2).
 */
export function saveAccepted(store: CallStore, subject: Study, text: string, checks: CaseCheck[], report: string,
  meta: Record<string, unknown> = {}, extraFiles: Record<string, string> = {}): { id: string; cases: number } | undefined {
  const accepted = checks.filter(check => check.accepted).map(check => check.position);
  if (!accepted.length) return undefined;
  const finalText = accepted.length === checks.length ? text : keepCases(text, accepted);
  const hashes = caseHashes(finalText);
  const kept = checks.filter(check => check.accepted);
  const record = [...subject.records.values()].at(-1)!;
  const links: { caseHash: string; callId: string; role: CaseRole; verdict?: Verdict | null }[] = [];
  kept.forEach((check, index) => {
    for (const result of check.results) if (result.split === 'training')
      links.push({ caseHash: hashes[index]!, callId: result.callId, role: 'training', verdict: result.verdict === 'error' || result.verdict === 'skipped' ? null : result.verdict });
  });
  const id = store.saveCompilation({ definitionKey: subject.key, definitionId: subject.definition.id, definitionName: subject.definition.name,
    definitionSource: subject.definition.source ?? null, interfaceHash: record.definition.interface, programRoot: record.program_root,
    files: { ...extraFiles, 'cases.ts': finalText, 'report.md': report, 'meta.json': JSON.stringify({ definition: subject.definition.name, key: subject.key,
      source: subject.definition.source ?? null, interface: record.definition.interface, calls: subject.examples.length,
      created: new Date().toISOString(), ...meta }, null, 2) + '\n' },
    caseHashes: hashes, links });
  kept.forEach((check, index) => {
    for (const result of check.results) if (result.split === 'held-out' && !['skipped'].includes(result.verdict))
      store.caseVerdict(hashes[index]!, result.callId, 'held-out', result.verdict === 'error' ? 'worse' : result.verdict as Verdict);
  });
  return { id, cases: kept.length };
}
