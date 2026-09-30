/**
 * The inline-natlang and observation-driven curriculum (docs/INLINE_NATLANG_TRAINING_DATA_PLAN.md).
 *
 * A curriculum case is an ordinary `natlang.program/2` record with a `curriculum` block. The block is
 * oracle metadata: the collector gives the model only `semantics`, so nothing here reaches the teacher.
 * It names the case's family and counterfactual group, the decisive observations that the opening
 * does not show, which techniques a reference demonstrates, and a reference solution that the
 * builder replays through the collector's execution path before the case is admitted as a seed.
 */
import type { ModelTurn, ModelTurnRequest } from '../contracts.js';
import { executeProgram, recordDigest, sha256, type ProgramRun } from './collector.js';
import { PROGRAM_VERSION, type ProgramRecord } from './program.js';
import { DATA_QUALITY_VERSION, csvRows } from './oracle.js';
import { trainingQualityReason, runtimeFailureReason, RETIRED_FAMILIES, quarantineReason } from './curriculum-policy.js';
import { sourceConversionProblems } from './source-conversion.js';

export const CURRICULUM_VERSION = 'natlang.inline_curriculum/1';
export const CURRICULUM_ADMISSION_VERSION = 'natlang.inline_curriculum_admission/2';

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
  /** Reference technique target for delegation; correct direct/delegated answers are admitted. */
  inline: 'required' | 'optional' | 'avoid';
  /** Whether a correct trajectory edits a callable function (edit_code): a real defect, or a correct helper. */
  edits?: 'required' | 'forbidden' | 'optional';
  /** A correct trajectory calls a named callable `.nl` function (the helper that fits), or runs `iterateOn`. */
  named?: 'required';
  iterate?: 'required';
  world_semantics?: 'open_world' | 'closed_world' | 'defeasible';
  /** Oracle provenance: world assertions, what must be retrieved, and background knowledge that bridges them. */
  evidence: { world: string[]; retrieved: string[]; background: string[] };
  assumptions: string[];
  decisive: Decisive[];
  /** Any one annotated supporting sentence is sufficient when the answer oracle also passes. */
  answer_evidence?: string[];
  /** For follow-up cases, the next actions that are plausible before the decisive observation. */
  plausible_actions: string[];
  minimum_sequence: string[];
  /** A replayable solution: root tool calls in order, and answers for child calls keyed by fragments of the child's opening (all must appear). */
  // A child answer with `call` answers with that tool call (such as `return_result` with status `blocked`) instead of a value; `calls` plays
  // several turns in order (for example an eval that acts, then return_result).
  /** failures: how many of the reference's actions meet the obstacle the case is about (a closed road, a locked card). */
  reference: { root: ReferenceCall[]; failures?: number; children?: { match: string | string[];
    /** Evidence shown by the scripted reference; task evidence contracts are checked separately. */
    evidence?: string[]; value?: unknown; call?: ReferenceCall; calls?: ReferenceCall[] }[] };
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
import { replacesPlantedFailure } from './seeded-failure.js';
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
  /** An eval read the call's transcript through, entry by entry in a loop, instead of searching it. */
  transcriptDump: boolean;
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
  let rootTurns = 0, evals = 0, edits = 0, functionEdits = 0, pageReads = 0, usesIterateOn = false, regexJudgment = false, transcriptDump = false, firstDecision = -1, finalTurn = -1, inlineCalls = 0, namedChildCalls = 0;
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
        if (/\b(?:for|while|forEach|map)\b[\s\S]{0,300}transcript\.entry\s*\(/.test(String(args.code ?? ''))) transcriptDump = true;
      }
      if (call.tool === 'edit_code') functionEdits++;
      if (call.tool === 'edit_code' || call.tool === 'edit_file' || call.tool === 'write_file') edits++;
      if (call.tool === 'read_page') pageReads++;
      if (firstDecision === -1 && (TERMINAL.has(call.tool) || (call.tool === 'eval' && stagesResult(String(args.code ?? '')) &&
          !evalFailed(trajectory, index, rootName))))
        firstDecision = index;
    }
  });
  if (firstDecision === -1) firstDecision = finalTurn;
  return { rootTurns, evals, edits, functionEdits, pageReads, usesIterateOn, regexJudgment, transcriptDump, inlineCalls, namedChildCalls, firstDecision, finalTurn, observedAt };
}

export type Admission = { id: string; program_id: string; admitted: boolean; reasons: string[];
  /** Observations that do not reject a row, such as a correct answer judged directly rather than inline. */
  notes: string[]; facts: RunFacts;
  family: string; slice: Slice; domain: Domain; mode: string; inline: string; pair_group: string | null;
  /** Strength of the case's answer oracle, when declared in program semantics. */
  oracle_level?: 'exact' | 'normalized' | 'span' | 'agreement' | 'judged' };

function oracleLevel(value: unknown): 'exact' | 'normalized' | 'span' | 'agreement' | 'judged' | undefined {
  const level = typeof value === 'string' ? value : value && typeof value === 'object' && !Array.isArray(value) ?
    (value as Record<string, unknown>).level : undefined;
  return ['exact', 'normalized', 'span', 'agreement', 'judged'].includes(String(level)) ?
    level as 'exact' | 'normalized' | 'span' | 'agreement' | 'judged' : undefined;
}

/**
 * Admission for one collected row: the collector's contract verdict plus the causal checks. A follow-up
 * case is admitted only when every decisive observation was visible before the root's first result
 * decision; technique differences are recorded as notes. The number of evals is never a criterion.
 */
/**
 * Outcomes the runtime no longer produces. A row that met one worked around a limitation a served model will not
 * meet, and its history shows the model something untrue about the runtime.
 */
const OBSOLETE_OUTCOMES: [string, RegExp][] = [
  ['inline_delegation_ban', /This call is itself a judgment handed over with nl, so make it here instead of handing it on/],
  ['duplicate_injected_binding', /invalid-binding: Injected binding .* is duplicated/],
  // nl results nothing typed run open now.
  ['nl_unknown_return', /nothing that uses it says what it should be/],
  // Parts of an nl parameter or result that nothing types are open.
  ['nl_untyped_part', /the type resolves to `any`|the type is unconstrained `unknown`/],
  // Any function runs with .iterateOn(initial), not only a natural-language one.
  ['iterate_method', /\.iterateOn is not a function/],
  // Eval awaits what it keeps, so no local holds a promise.
  ['unawaited_promise', /\[Promise #\d+; live value/],
  // nl called like a function is the one-shot call it reads as.
  ['nl_call_arity', /nl@eval:\d+ expects \d+ arguments/],
  // Compaction keeps the opening, the note and the latest exchange instead of eliding outputs.
  ['elided_output', /elided to keep this conversation within its context budget/],
  // A request the server refuses as too long is retried, compacted.
  ['context_exceeded', /Context size has been exceeded/],
];

/** The obsolete outcomes a trajectory's tool results show, including a type an eval declared and then could not use. */
export function obsoleteOutcomes(trajectory: Turn[]): string[] {
  const found = new Set<string>();
  for (const turn of trajectory) {
    const context = turn.context ?? [];
    const code = context.flatMap(message => (message.tool_calls ?? []).map(call => call.function?.arguments ?? '')).join('\n');
    for (const message of context) {
      if (message.role !== 'tool') continue;
      const content = text(message.content);
      for (const [name, pattern] of OBSOLETE_OUTCOMES) if (pattern.test(content)) found.add(name);
      for (const [, type] of content.matchAll(/unknown type name (\w+)/g))
        if (new RegExp(`\\b(?:type|interface)\\s+${type}\\b`).test(code)) found.add('eval_declared_type');
    }
  }
  return [...found];
}

/** Whether text shows a marker: as it is, or escaped inside JSON (a quote in a tool result shown as data). */
const shows = (text: string, marker: string) => text.includes(marker) || text.includes(JSON.stringify(marker).slice(1, -1));

/** Identify an item by the child's instructions and arguments, excluding unrelated captured collections. */
function childIdentity(context: Message[]): string {
  const opening = context.slice(1, openingLength(context)).map(message => ({ ...message,
    tool_calls: message.tool_calls?.map(call => {
      const args = call.function?.arguments;
      if (!args) return call;
      try {
        const parsed = JSON.parse(args);
        if (typeof parsed.code === 'string' && parsed.code.includes('const inputs = read_inputs();')) parsed.code = parsed.code.split('// Variables of the calling code, captured by this call:')[0];
        return { ...call, function: { ...call.function, arguments: JSON.stringify(parsed) } };
      } catch { return call; }
    }),
  }));
  return opening.map(message => text(message.content) +
    (message.tool_calls ?? []).map(call => call.function?.arguments ?? '').join('\n')).join('\n');
}

export function admitRow(row: { id?: string; task: { program_ir: ProgramRecord }; provenance?: Record<string, unknown>; outcome?: Record<string, unknown>;
  trajectory?: unknown[] }): Admission {
  const record = row.task.program_ir as CurriculumRecord, c = record.curriculum;
  const facts = runFacts(record, (row.trajectory ?? []) as Turn[]);
  const reasons: string[] = sourceConversionProblems(row), notes: string[] = [];
  if (RETIRED_FAMILIES.has(c.family)) reasons.push('retired_family');
  const quarantine = quarantineReason(record) ?? trainingQualityReason(row) ?? runtimeFailureReason(row);
  if (quarantine) reasons.push(quarantine);
  for (const name of obsoleteOutcomes((row.trajectory ?? []) as Turn[])) reasons.push(`obsolete_outcome:${name}`);
  const outcome = row.outcome ?? {};
  if (!['done', 'quiesced', 'failed'].includes(String(outcome.status))) reasons.push('incomplete_trajectory');
  else if (!outcome.accepted && (outcome.quality_pending as unknown[] | undefined)?.length) reasons.push('quality_pending');
  else if (!outcome.accepted) reasons.push(record.semantics.operation === 'blocked' && outcome.status === 'done' ?
    'fabricated_result' : 'wrong_return');
  for (const item of c.decisive) {
    // A compile failure prevented by replacing its planted first action is not an observation prerequisite.
    // Runtime failures may expose state/effects, so their observation requirements remain intact.
    if (item.source === 'error' && record.semantics.failure_seed?.kind === 'compile' &&
      replacesPlantedFailure(record) && (outcome.seeded_failure as { replaced_by_handoff?: boolean } | undefined)?.replaced_by_handoff) {
      if (!notes.includes('planted_compile_failure_prevented')) notes.push('planted_compile_failure_prevented');
      continue;
    }
    const at = facts.observedAt[item.marker]!;
    if (at === -1) reasons.push(`missing_observation:${item.marker}`);
    else if (c.mode === 'followup' && at > facts.firstDecision) reasons.push(`premature_choice:${item.marker}`);
  }
  const allObserved = ((row.trajectory ?? []) as Turn[]).flatMap(turn => (turn.context ?? [])
    .filter(message => message.role === 'tool').map(message => text(message.content))).join('\n');
  if (c.answer_evidence?.length && !c.answer_evidence.some(marker => shows(allObserved, marker)))
    reasons.push('missing_answer_evidence');
  const filesSpec = record.semantics.files_oracle;
  if (['rewrite', 'csv', 'counts', 'json-string-record', 'tatqa-answer-record'].includes(filesSpec?.compare ?? '') &&
      (outcome.files_check as { quality_version?: number } | undefined)?.quality_version !== DATA_QUALITY_VERSION)
    reasons.push('unreviewed_files_oracle');
  if (filesSpec?.quote_sources) {
    try {
      const report = (outcome.files as Record<string, string> | undefined)?.[filesSpec.report ?? 'clauses.csv'];
      for (const row of csvRows(report ?? '').slice(1)) if (row[1]?.trim() && !shows(allObserved, row[1]))
        reasons.push(`unobserved_output_quote:${row[0]}`);
    } catch { reasons.push('invalid_output_report'); }
  }
  for (const child of c.reference.children ?? []) {
    const matches = Array.isArray(child.match) ? child.match : [child.match];
    // A child is the item's by what it was given: the fragments naming it, or else the item's evidence (a child given
    // an item's text instead of its file sees the text, not the file's name).
    const turns = ((row.trajectory ?? []) as Turn[]).filter(turn => {
      if (callName(turn.context ?? []) === record.semantics.root.replace(/\.nl$/, '').split('/').pop()) return false;
      const opening = childIdentity(turn.context ?? []);
      return matches.every(fragment => opening.includes(fragment)) ||
        !!child.evidence?.length && child.evidence.every(marker => shows(opening, marker));
    });
    // Reference children demonstrate one solution; mixed direct/delegated coverage is valid.
    const firstAnswer = turns.find(turn => (turn.assistant?.calls ?? []).some(call =>
      call.tool === 'return_result' || call.tool === 'eval' && stagesResult(String((call.arguments as Record<string, unknown>)?.code ?? ''))) ||
      Boolean(turn.assistant?.content?.trim()));
    if (firstAnswer) {
      const seen = childIdentity(firstAnswer.context ?? []) + '\n' + (firstAnswer.context ?? [])
        .filter(message => message.role === 'tool').map(message => text(message.content)).join('\n');
      for (const marker of child.evidence ?? []) if (!shows(seen, marker) && !shows(allObserved, marker))
        if (!notes.includes('reference_evidence_differs')) notes.push('reference_evidence_differs');
    }
  }
  // A correct answer judged directly is a fine sample; a keyword or regex stand-in for a judgment is not.
  if (c.inline === 'required' && !facts.inlineCalls) {
    notes.push('judged_directly');
  }
  if (c.inline === 'avoid' && facts.inlineCalls) notes.push('delegated_optional');
  // Technique/efficiency differences describe coverage; they do not establish an incorrect result.
  if (facts.transcriptDump) notes.push('transcript_dump');
  if (facts.regexJudgment) notes.push('regex_used');
  if (c.edits === 'required' && !facts.functionEdits) reasons.push('defect_not_repaired');
  if (c.edits === 'forbidden' && facts.functionEdits) reasons.push('unwarranted_edit');
  if (c.named === 'required' && !facts.namedChildCalls) notes.push('named_helper_unused');
  if (c.iterate === 'required' && !facts.usesIterateOn) notes.push('iterate_missing');
  return { id: String(row.id ?? record.id), program_id: record.id, admitted: !reasons.length, reasons, notes, facts,
    family: c.family, slice: c.slice, domain: c.domain, mode: c.mode, inline: c.inline, pair_group: c.pair_group,
    ...(oracleLevel(record.semantics.oracle) ? { oracle_level: oracleLevel(record.semantics.oracle) } : {}) };
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

/**
 * A case's reference solution as a model: the root call's scripted actions in order, and each child call's scripted
 * answer, found by the fragments of its opening. Each action comes with a short line saying what it does
 * (actionNote), in place of the reasoning a model would give; it does not argue for a result.
 */
export function referenceDriver(record: CurriculumRecord): (request: ModelTurnRequest) => Promise<ModelTurn> {
  const rootName = record.semantics.root.replace(/\.nl$/, '').split('/').pop()!;
  let step = 0, seeded = !record.semantics.failure_seed;
  const childTurns = new Map<string, number>();
  return async request => {
    const context = request.messages as Message[], name = callName(context);
    let calls: [string, Record<string, unknown>][];
    if (name === rootName && !seeded) {
      // The collector answers a seeded program's first root turn with the failing eval; so does the replay.
      seeded = true;
      calls = [['eval', { code: record.semantics.failure_seed!.code }]];
    } else if (name === rootName) {
      const call = record.curriculum.reference.root[step++];
      // Reference-only directive: finish a computed, staged value through the ordinary reply path.
      // This is not a runtime tool and must never be emitted as a tool call.
      if (call?.[0] === 'reply') return { text: String(call[1].text ?? ''), reasoning: 'Finish the staged result.' };
      calls = [call ? [call[0], call[1]] :
        ['return_result', { status: 'failed', reason: 'The reference solution ended without finishing the call.' }]];
    } else {
      const opening = openingText(context);
      const answer = record.curriculum.reference.children?.find(child =>
        (Array.isArray(child.match) ? child.match : [child.match]).every(fragment => opening.includes(fragment)));
      const turn = childTurns.get(opening) ?? 0;
      childTurns.set(opening, turn + 1);
      const scripted = answer?.calls?.[turn];
      calls = [scripted ? [scripted[0], scripted[1]] :
        answer ? (answer.call ? [answer.call[0], answer.call[1]] : ['return_result', { status: 'success', value: answer.value }]) :
        ['return_result', { status: 'failed', reason: `The reference has no answer for the child call ${name}.` }]];
    }
    return { calls, reasoning: actionNote(calls) };
  };
}

/** One line on what a scripted action does, from the action alone. */
export function actionNote(calls: [string, Record<string, unknown>][]): string {
  return calls.map(([tool, args]) => {
    const code = String(args.code ?? '');
    switch (tool) {
      case 'eval':
        if (/\biterateOn\s*\(/.test(code)) return 'I run the step with iterateOn until it is finished.';
        if (/\bnl\s*(?:<[^`]*>)?`/.test(code)) return 'I hand the judgment for each item to nl and collect the results.';
        if (/\.pages?\s*\(/.test(code)) return 'I read the pages of the data.';
        if (/^\s*return\b/m.test(code)) return 'I compute the result in code and return it.';
        return 'I run the next step in eval.';
      case 'read_code': return `I read the code of ${String(args.name ?? 'the function')}.`;
      case 'edit_code': return `I fix ${String(args.name ?? 'the function')}.`;
      case 'read_page': return 'I read the next page of that output.';
      case 'return_result': return args.status === 'success' ? 'I return the result.' :
        `I finish the call with status ${String(args.status)}.`;
      default: return `I use ${tool}.`;
    }
  }).join(' ');
}

/** Replay a case's reference solution through the collector's execution path, recording a trajectory. */
export async function replayReference(record: CurriculumRecord, systemPrompt: string):
    Promise<{ run: ProgramRun; trajectory: Turn[] }> {
  const trajectory: Turn[] = [], reference = referenceDriver(record);
  const driver = async (request: ModelTurnRequest): Promise<ModelTurn> => {
    const response = await reference(request);
    trajectory.push({ context: structuredClone(request.messages) as Message[],
      assistant: { calls: (response.calls ?? []).map(([tool, args]) => ({ tool, arguments: args })) } });
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
      if (admission.notes.includes('judged_directly')) problems.push('reference does not use the technique its case requires');
      // Every scripted action runs, except a seeded failure's own eval and the obstacles the reference declares.
      const failed = ((run.outcome.action_ledger ?? []) as Record<string, unknown>[]).filter(event => ['rejected', 'refused', 'error'].includes(String(event.outcome)));
      const expectedFailures = (record.semantics.failure_seed ? 1 : 0) + (record.curriculum.reference.failures ?? 0);
      if (failed.length !== expectedFailures)
        problems.push(`${failed.length} reference actions failed where ${expectedFailures} should: ${failed.map(event => `${String(event.name)}: ${String(event.result_text ?? '').slice(0, 300)}`).join('; ')}`);
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
