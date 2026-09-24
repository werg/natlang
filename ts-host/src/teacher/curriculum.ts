/**
 * The inline-natlang and observation-driven curriculum (docs/INLINE_NATLANG_TRAINING_DATA_PLAN.md).
 *
 * A curriculum case is an ordinary `natlang.program/2` record with a `curriculum` block. The block is
 * oracle metadata: the collector gives the model only `semantics`, so nothing here reaches the teacher.
 * It names the case's family and counterfactual group, the decisive observations that the opening
 * does not show, whether an inline `nl` is required or gratuitous, and a reference solution that the
 * builder replays through the collector's execution path before the case is admitted as a seed.
 */
import type { ModelTurn, ModelTurnRequest } from '../contracts.js';
import { executeProgram, recordDigest, sha256, type ProgramRun } from './collector.js';
import { PROGRAM_VERSION, type ProgramRecord } from './program.js';

export const CURRICULUM_VERSION = 'natlang.inline_curriculum/1';
export const CURRICULUM_ADMISSION_VERSION = 'natlang.inline_curriculum_admission/1';

export const SLICES = ['inline_placement', 'observation_followup', 'nested_scoped', 'iterate', 'folder_failure'] as const;
export const DOMAINS = ['logic', 'relational', 'actor', 'other'] as const;
/** Target shares from the plan, measured over admitted cases. */
export const SLICE_TARGETS: Record<Slice, number> = { inline_placement: 0.25, observation_followup: 0.30,
  nested_scoped: 0.15, iterate: 0.15, folder_failure: 0.15 };
export const DOMAIN_TARGETS: Record<Domain, number> = { logic: 0.30, relational: 0.25, actor: 0.20, other: 0.25 };
export type Slice = typeof SLICES[number];
export type Domain = typeof DOMAINS[number];

/** Where a decisive observation first becomes visible to the model. */
export type ObservationSource = 'page' | 'eval' | 'function_source' | 'file' | 'child' | 'error';
export type Decisive = { marker: string; source: ObservationSource; note: string };
export type ReferenceCall = [tool: string, args: Record<string, unknown>];

export type Curriculum = {
  version: typeof CURRICULUM_VERSION;
  family: string;
  family_version: number;
  shape: string;
  variant: string;
  /** Variants that share one visible request and differ in a hidden observation; their openings must be identical. */
  pair_group: string | null;
  /** Every variant of one underlying problem shares a split group. */
  split_group: string;
  slice: Slice;
  domain: Domain;
  /** `single_call`: one well-formed eval can be right. `followup`: a later choice depends on an observation. */
  mode: 'single_call' | 'followup';
  /** Whether a correct trajectory creates an inline `nl`: needed, allowed, or a gratuitous child. */
  inline: 'required' | 'optional' | 'avoid';
  /** Whether a correct trajectory edits a callable function (edit_function): a real defect, or a correct helper. */
  edits?: 'required' | 'forbidden' | 'optional';
  /** A correct trajectory calls a named callable `.nl` function (the helper that fits), or runs `iterateOn`. */
  named?: 'required';
  iterate?: 'required';
  world_semantics?: 'open_world' | 'closed_world' | 'defeasible';
  /** Oracle provenance: world assertions, what must be retrieved, and background knowledge that bridges them. */
  evidence: { world: string[]; retrieved: string[]; background: string[] };
  assumptions: string[];
  decisive: Decisive[];
  /** For follow-up cases, the next actions that are plausible before the decisive observation. */
  plausible_actions: string[];
  minimum_sequence: string[];
  /** A replayable solution: root tool calls in order, and answers for child calls keyed by fragments of the child's opening (all must appear). */
  // A child answer with `call` answers with that tool call (such as `return_result` with status `blocked`) instead of a value; `calls` plays
  // several turns in order (for example an eval that acts, then return_result).
  reference: { root: ReferenceCall[]; children?: { match: string | string[]; value?: unknown; call?: ReferenceCall; calls?: ReferenceCall[] }[] };
};
export type CurriculumRecord = ProgramRecord & { curriculum: Curriculum; family: string; split: string };

const fail = (id: string, message: string): never => { throw new Error(`${id}: ${message}`); };

export function validateCurriculum(record: CurriculumRecord): void {
  const id = record.id, c = record.curriculum;
  if (record.version !== PROGRAM_VERSION) fail(id, 'not natlang.program/2');
  if (!c || c.version !== CURRICULUM_VERSION) fail(id, 'missing curriculum block');
  if (!SLICES.includes(c.slice)) fail(id, `unknown slice ${c.slice}`);
  if (!DOMAINS.includes(c.domain)) fail(id, `unknown domain ${c.domain}`);
  if (!['single_call', 'followup'].includes(c.mode)) fail(id, `unknown mode ${c.mode}`);
  if (!['required', 'optional', 'avoid'].includes(c.inline)) fail(id, `unknown inline mode ${c.inline}`);
  if (c.edits !== undefined && !['required', 'forbidden', 'optional'].includes(c.edits)) fail(id, `unknown edits mode ${c.edits}`);
  if (c.named !== undefined && c.named !== 'required') fail(id, `unknown named mode ${c.named}`);
  if (c.iterate !== undefined && c.iterate !== 'required') fail(id, `unknown iterate mode ${c.iterate}`);
  if (!c.family || !c.shape || !c.variant || !c.split_group) fail(id, 'family, shape, variant, and split group are required');
  if (c.mode === 'followup') {
    if (!c.decisive.length) fail(id, 'a follow-up case needs a decisive observation');
    if (c.plausible_actions.length < 2) fail(id, 'a follow-up case needs two plausible actions before its observation');
  }
  for (const item of c.decisive) if (!item.marker.trim() || item.marker.length < 4)
    fail(id, `decisive marker ${JSON.stringify(item.marker)} is too short to be unambiguous`);
  if (!c.reference?.root?.length) fail(id, 'a reference solution is required');
}

// ---- Trajectory reading ----------------------------------------------------------------------

import { openingLength, openingText, text, type Message } from './opening.js';
type Turn = { context: Message[]; assistant?: { calls?: { tool: string; arguments: unknown }[]; content?: string } };

/** The name of the call a request belongs to, from its opening line. */
export function callName(context: Message[]): string {
  const match = /^You are inside this call: ([^(]+)\(/.exec(String(context[1]?.content ?? ''));
  return match?.[1] ?? '';
}
export { openingLength } from './opening.js';
const isInline = (name: string) => name.startsWith('nl@');
const TERMINAL = new Set(['return_result']);
/** A top-level `return` in an eval stages a result: a decision as much as return_result is. */
/** Whether the root's eval at this turn failed: its result (shown in the root's next turn) kept nothing. */
function evalFailed(trajectory: Turn[], index: number, rootName: string): boolean {
  const next = trajectory.slice(index + 1).find(turn => callName(turn.context ?? []) === rootName);
  return !!next && text(next.context.at(-1)?.content).includes('Nothing else from this eval was kept.');
}
const stagesResult = (code: string) => /^return\b/m.test(code) || /\breturn_result\s*\(/.test(code);

export type RunFacts = {
  rootTurns: number; evals: number; edits: number; functionEdits: number; pageReads: number; usesIterateOn: boolean; inlineCalls: number; namedChildCalls: number;
  /** An eval tested text against a regular expression with alternatives: a keyword stand-in for a judgment. */
  regexJudgment: boolean;
  /** Index (in the flat trajectory) of the root's first result decision, and of its final turn. */
  firstDecision: number; finalTurn: number;
  /** For each decisive marker, the first trajectory index whose request shows it outside the root opening, or -1. */
  observedAt: Record<string, number>;
};

/** Facts about a collected trajectory that the causal checks need. */
export function runFacts(record: CurriculumRecord, trajectory: Turn[]): RunFacts {
  const rootName = record.semantics.root.replace(/\.nl$/, '').split('/').pop()!;
  const observedAt: Record<string, number> = Object.fromEntries(record.curriculum.decisive.map(item => [item.marker, -1]));
  const children = new Set<string>();
  let rootTurns = 0, evals = 0, edits = 0, functionEdits = 0, pageReads = 0, usesIterateOn = false, regexJudgment = false, firstDecision = -1, finalTurn = -1, inlineCalls = 0, namedChildCalls = 0;
  trajectory.forEach((turn, index) => {
    const context = turn.context ?? [], name = callName(context), root = name === rootName;
    // Observations: anything a tool showed, in this call or a child the model delegated to, past the root opening.
    const visible = context.slice(root ? openingLength(context) : 1).filter(message => message.role === 'tool' || !root)
      .map(message => text(message.content) + (message.tool_calls ?? []).map(call => call.function?.arguments ?? '').join(''))
      .join('\n');
    for (const marker of Object.keys(observedAt)) if (observedAt[marker] === -1 && visible.includes(marker)) observedAt[marker] = index;
    if (!root) {
      const key = openingText(context);
      if (!children.has(key)) { children.add(key); if (isInline(name)) inlineCalls++; else namedChildCalls++; }
      return;
    }
    rootTurns++; finalTurn = index;
    for (const call of turn.assistant?.calls ?? []) {
      const args = (call.arguments ?? {}) as Record<string, unknown>;
      if (call.tool === 'eval') {
        evals++;
        if (/\.iterateOn\s*\(|\biterateOn\s*\(/.test(String(args.code ?? ''))) usesIterateOn = true;
        if (/\/[^/\n]*\w+\|\w+[^/\n]*\/[gimsuy]*\.test\s*\(/.test(String(args.code ?? ''))) regexJudgment = true;
      }
      if (call.tool === 'edit_function') functionEdits++;
      if (call.tool === 'edit_function' || call.tool === 'edit_file' || call.tool === 'write_file') edits++;
      if (call.tool === 'read_page') pageReads++;
      if (firstDecision === -1 && (TERMINAL.has(call.tool) || (call.tool === 'eval' && stagesResult(String(args.code ?? '')) &&
          !evalFailed(trajectory, index, rootName))))
        firstDecision = index;
    }
  });
  if (firstDecision === -1) firstDecision = finalTurn;
  return { rootTurns, evals, edits, functionEdits, pageReads, usesIterateOn, regexJudgment, inlineCalls, namedChildCalls, firstDecision, finalTurn, observedAt };
}

export type Admission = { id: string; program_id: string; admitted: boolean; reasons: string[];
  /** Observations that do not reject a row, such as a correct answer judged directly rather than inline. */
  notes: string[]; facts: RunFacts;
  family: string; slice: Slice; domain: Domain; mode: string; inline: string; pair_group: string | null };

/**
 * Admission for one collected row: the collector's contract verdict plus the causal checks. A follow-up
 * case is admitted only when every decisive observation was visible before the root's first result
 * decision; the inline mode requires or forbids an inline child. The number of evals is never a criterion.
 */
export function admitRow(row: { id?: string; task: { program_ir: ProgramRecord }; outcome?: Record<string, unknown>;
  trajectory?: unknown[] }): Admission {
  const record = row.task.program_ir as CurriculumRecord, c = record.curriculum;
  const facts = runFacts(record, (row.trajectory ?? []) as Turn[]);
  const reasons: string[] = [];
  const outcome = row.outcome ?? {};
  if (!['done', 'quiesced', 'failed'].includes(String(outcome.status))) reasons.push('incomplete_trajectory');
  else if (!outcome.accepted) reasons.push(record.semantics.operation === 'blocked' && outcome.status === 'done' ?
    'fabricated_result' : 'wrong_return');
  for (const item of c.decisive) {
    const at = facts.observedAt[item.marker]!;
    if (at === -1) reasons.push(`missing_observation:${item.marker}`);
    else if (c.mode === 'followup' && at > facts.firstDecision) reasons.push(`premature_choice:${item.marker}`);
  }
  // A correct answer judged directly is a fine sample; a keyword or regex stand-in for a judgment is not.
  const notes: string[] = [];
  if (c.inline === 'required' && !facts.inlineCalls) {
    if (facts.regexJudgment) reasons.push('regex_judgment'); else notes.push('judged_directly');
  }
  if (c.inline === 'avoid' && facts.inlineCalls) reasons.push('gratuitous_inline');
  if (c.edits === 'required' && !facts.functionEdits) reasons.push('defect_not_repaired');
  if (c.edits === 'forbidden' && facts.functionEdits) reasons.push('unwarranted_edit');
  if (c.named === 'required' && !facts.namedChildCalls) reasons.push('named_helper_unused');
  if (c.iterate === 'required' && !facts.usesIterateOn) reasons.push('iterate_missing');
  return { id: String(row.id ?? record.id), program_id: record.id, admitted: !reasons.length, reasons, notes, facts,
    family: c.family, slice: c.slice, domain: c.domain, mode: c.mode, inline: c.inline, pair_group: c.pair_group };
}

// ---- Build-time verification -----------------------------------------------------------------

const REFERENCE_OPTIONS = { contextTokens: 16384, maxTurns: 60, rootSeed: 0, runId: 'curriculum-reference' };

/** The opening the model would see (user message and pre-filled scope exchanges), without running a model. */
export async function renderOpening(record: ProgramRecord, systemPrompt: string): Promise<string> {
  let opening = '';
  const driver = async (request: ModelTurnRequest): Promise<ModelTurn> => {
    if (!opening) opening = openingText(request.messages as Message[]);
    return { calls: [['return_result', { status: 'failed', reason: 'The opening render stops before the first model turn.' }]] };
  };
  await executeProgram(record, driver, { ...REFERENCE_OPTIONS, systemPrompt });
  return opening;
}

/** Replay a case's reference solution through the collector's execution path, recording a trajectory. */
export async function replayReference(record: CurriculumRecord, systemPrompt: string):
    Promise<{ run: ProgramRun; trajectory: Turn[] }> {
  const rootName = record.semantics.root.replace(/\.nl$/, '').split('/').pop()!;
  const trajectory: Turn[] = [];
  let step = 0, seeded = !record.semantics.failure_seed;
  const childTurns = new Map<string, number>();
  const driver = async (request: ModelTurnRequest): Promise<ModelTurn> => {
    const context = structuredClone(request.messages) as Message[], name = callName(context);
    let response: ModelTurn;
    if (name === rootName && !seeded) {
      // The collector answers a seeded program's first root turn with the failing eval; so does the replay.
      seeded = true;
      response = { calls: [['eval', { code: record.semantics.failure_seed!.code }]] };
    } else if (name === rootName) {
      const call = record.curriculum.reference.root[step++];
      response = call ? { calls: [[call[0], call[1]]] } :
        { calls: [['return_result', { status: 'failed', reason: 'The reference solution ended without finishing the call.' }]] };
    } else {
      const opening = openingText(context);
      const answer = record.curriculum.reference.children?.find(child =>
        (Array.isArray(child.match) ? child.match : [child.match]).every(fragment => opening.includes(fragment)));
      const turn = childTurns.get(opening) ?? 0;
      childTurns.set(opening, turn + 1);
      const scripted = answer?.calls?.[turn];
      response = scripted ? { calls: [[scripted[0], scripted[1]]] } :
        answer ? { calls: [answer.call ? [answer.call[0], answer.call[1]] : ['return_result', { status: 'success', value: answer.value }]] } :
        { calls: [['return_result', { status: 'failed', reason: `The reference has no answer for the child call ${name}.` }]] };
    }
    trajectory.push({ context, assistant: { calls: (response.calls ?? []).map(([tool, args]) => ({ tool, arguments: args })) } });
    return response;
  };
  const run = await executeProgram(record, driver, { ...REFERENCE_OPTIONS, systemPrompt });
  return { run, trajectory };
}

export type Verification = { id: string; ok: boolean; problems: string[]; opening_sha256: string };

/**
 * Verify a batch before it becomes seeds: each case validates, its decisive markers are not in its
 * opening, its reference replays to an admitted result, and each counterfactual group shares one
 * opening while its expected results differ.
 */
export async function verifyCases(records: CurriculumRecord[], systemPrompt: string): Promise<Verification[]> {
  const results: Verification[] = [];
  const groups = new Map<string, { opening: string; expected: string[]; ids: string[] }>();
  for (const record of records) {
    const problems: string[] = [];
    let opening = '';
    try {
      validateCurriculum(record);
      opening = await renderOpening(record, systemPrompt);
      for (const item of record.curriculum.decisive)
        if (opening.includes(item.marker)) problems.push(`decisive marker ${JSON.stringify(item.marker)} is visible in the opening`);
      const { run, trajectory } = await replayReference(record, systemPrompt);
      const admission = admitRow({ task: { program_ir: record }, outcome: run.outcome, trajectory });
      if (!admission.admitted) problems.push(`reference not admitted: ${admission.reasons.join(', ')}` +
        (run.outcome.accepted ? '' : ` (status ${String(run.outcome.status)}, value ${JSON.stringify(run.outcome.value)}, detail ${JSON.stringify(run.outcome.detail)})`));
    } catch (error) { problems.push(error instanceof Error ? error.message : String(error)); }
    const group = record.curriculum?.pair_group;
    if (group) {
      const entry = groups.get(group) ?? { opening, expected: [], ids: [] };
      if (entry.ids.length && entry.opening !== opening) problems.push(`opening differs from ${entry.ids[0]} in pair group ${group}`);
      entry.expected.push(JSON.stringify([record.semantics.operation ?? null, record.semantics.expected, record.curriculum.edits ?? null]));
      entry.ids.push(record.id);
      groups.set(group, entry);
    }
    results.push({ id: record.id, ok: !problems.length, problems, opening_sha256: sha256(opening) });
  }
  for (const [group, entry] of groups) {
    if (entry.ids.length < 2) continue;
    if (new Set(entry.expected).size < 2) for (const result of results) if (entry.ids.includes(result.id)) {
      result.ok = false; result.problems.push(`pair group ${group} has one expected result for every variant`);
    }
  }
  return results;
}

// ---- Coverage --------------------------------------------------------------------------------

export type Coverage = { total: number; admitted: number; by: Record<string, Record<string, { total: number; admitted: number }>>;
  rejections: Record<string, number>; shares: { slice: Record<string, number>; domain: Record<string, number> } };

export function coverage(admissions: Admission[]): Coverage {
  const by: Coverage['by'] = {}, rejections: Record<string, number> = {};
  const bump = (axis: string, key: string, admitted: boolean) => {
    const entry = ((by[axis] ??= {})[key] ??= { total: 0, admitted: 0 });
    entry.total++; if (admitted) entry.admitted++;
  };
  for (const item of admissions) {
    for (const [axis, key] of [['family', item.family], ['slice', item.slice], ['domain', item.domain], ['mode', item.mode],
      ['inline', item.inline], ['root_turns', String(Math.min(item.facts.rootTurns, 8))],
      ['inline_calls', String(Math.min(item.facts.inlineCalls, 4))], ['iterate_on', String(item.facts.usesIterateOn)]] as const) bump(axis, key, item.admitted);
    for (const reason of item.reasons) { const key = reason.split(':')[0]!; rejections[key] = (rejections[key] ?? 0) + 1; }
  }
  const admitted = admissions.filter(item => item.admitted);
  const share = (axis: 'slice' | 'domain') => Object.fromEntries(Object.entries(by[axis] ?? {})
    .map(([key, value]) => [key, admitted.length ? Math.round(1000 * value.admitted / admitted.length) / 1000 : 0]));
  return { total: admissions.length, admitted: admitted.length, by, rejections, shares: { slice: share('slice'), domain: share('domain') } };
}

export const curriculumDigest = (record: CurriculumRecord) => recordDigest(record);
