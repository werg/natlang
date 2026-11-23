/**
 * Workspace combinators over folder transactions (plans/semantic-fp/03_SEMANTIC_STDLIB_AND_PATTERNS.md S2;
 * ts-host/std/DECOMPOSITION.md): `proposeAndVerify` (propose, verify exactly, re-propose on the failed draft with the
 * problems, accept only what passes) and `branchAndSelect` (several drafts in private copies, the passing ones judged
 * against a goal, one accepted). Nothing reaches the folder except through `accept`; the verifier is the caller's.
 */
import { APPLY_TO_FOLDER, type FolderHandle, type FolderProposal } from '../native/scoped-fs.js';
import { decideInFrame } from '../runtime/runtime.js';
import { semanticRepair, type Problems } from './converge.js';
import { beats as shippedBeats } from './judgments.js';
import { StdSpan, stdError } from './trace.js';

type Maybe<T> = T | Promise<T>;
/** A folder that proposes and accepts: a `Folder` or a `FolderHandle`. */
export type Workspace = {
  propose<R = unknown>(reducer: unknown, ...args: unknown[]): Promise<FolderProposal<R>>;
  accept<R>(proposal: FolderProposal<R>): Promise<unknown>;
};
/** A directory reducer: a TypeScript function `(folder, ...args)` or a `.nl` function with `kind: directory-reducer`. */
export type Reducer = unknown;

function problemsOf(found: unknown): string[] {
  if (!Array.isArray(found) || found.some(item => typeof item !== 'string'))
    throw stdError('std-check-result', 'verify returns the list of problems it found in the proposal, as strings, and an empty list when it passes.');
  return [...found];
}
function workspaceOf(folder: unknown, combinator: string): Workspace {
  const candidate = folder as Partial<Workspace> | undefined;
  if (typeof candidate?.propose !== 'function' || typeof candidate.accept !== 'function')
    throw stdError('std-arguments', `${combinator} works on a Folder or FolderHandle with write access, given as its first argument.`);
  return candidate as Workspace;
}

// --- proposeAndVerify -----------------------------------------------------------------------------------------------

export type ProposeOptions<R> = {
  /** The exact verifier, a service the reducer cannot edit: the problems it finds in a proposal, none when it passes. */
  verify: (proposal: FolderProposal<R>) => Maybe<Problems>;
  /** The most re-proposals after the first. */
  attempts: number;
  /** The reducer's arguments before `problems`, which is always its last argument (empty on the first proposal). */
  args?: unknown[];
};
export type Proposed<R> = {
  /** Whether a proposal passed and was accepted into the folder. */
  accepted: boolean;
  /** The accepted proposal, or else the one with the fewest problems (never installed). */
  proposal: FolderProposal<R>;
  /** The reducer's result for that proposal. */
  value: R;
  problems: string[];
  attempts: number;
  /** The verifier's problems with each proposal in order. */
  history: string[][];
};

/**
 * Propose a change with `reducer` on a private copy of `folder`, verify it, and accept it when it passes. A failing
 * proposal is proposed again from its own draft, with the verifier's problems as the reducer's last argument.
 */
export async function proposeAndVerify<R = unknown>(folder: Workspace, reducer: Reducer, options: ProposeOptions<R>): Promise<Proposed<R>> {
  const workspace = workspaceOf(folder, 'proposeAndVerify');
  if (typeof options?.verify !== 'function')
    throw stdError('std-arguments', 'proposeAndVerify(folder, reducer, { verify, attempts }) needs an exact verify function.');
  const args = options.args ?? [];
  const span = new StdSpan('proposeAndVerify');
  return span.around(async () => {
    // A reducer applied to the proposal's draft itself (no transaction of its own): the next draft continues the failed
    // one, keeping its changes, and the caller's reducer runs once in it with the problems the verifier found.
    const propose = (problems: string[], previous?: FolderProposal<R>) => span.run(() => workspace.propose<R>({
      [APPLY_TO_FOLDER]: async (draft: FolderHandle) => {
        if (previous) await draft.select(previous.folder);
        return draft.apply(reducer, ...args, problems);
      } }));
    const first = await propose([]);
    span.stage('propose', { attempt: 0 });
    const repair = await semanticRepair(first, { check: async proposal => problemsOf(await span.run(() => options.verify(proposal))),
      fix: (proposal, problems) => propose(problems, proposal), attempts: options.attempts,
      identity: proposal => proposal.folder.digest });
    span.stage('verify', { attempts: repair.attempts, ok: repair.ok });
    if (repair.ok) await span.run(() => workspace.accept(repair.value));
    span.stage('accept', { accepted: repair.ok });
    return { accepted: repair.ok, proposal: repair.value, value: repair.value.value, problems: repair.problems,
      attempts: repair.attempts, history: repair.history.map(attempt => attempt.problems) };
  }, result => result.accepted ? `accepted after ${result.attempts} re-proposals` : `not accepted: ${result.problems.length} problems`);
}

// --- branchAndSelect ------------------------------------------------------------------------------------------------

/** One branch: a reducer, or a reducer with its arguments. */
export type Branch = Reducer | { reducer: Reducer; args?: unknown[] };
/** A branch's proposal as the selection sees it. */
export type Candidate<R> = { index: number; proposal: FolderProposal<R>; value: R; problems: string[] };
export type SelectOptions<R> = {
  /** The exact verifier; a branch whose proposal has problems is not selected. */
  verify?: (proposal: FolderProposal<R>) => Maybe<Problems>;
  /** What the change is for. With two or more passing branches, the shipped `beats` judges them pairwise against it. */
  goal?: string;
  /** The caller's choice among the passing candidates instead: the `index` of the one to accept. */
  select?: (candidates: Candidate<R>[]) => Maybe<number>;
  /** A replacement for the shipped `beats(goal, first, second)`, such as `beats.on({ model })`. */
  beats?: (goal: string, first: unknown, second: unknown) => Promise<boolean>;
};
export type Selected<R> = {
  /** The index of the accepted branch, or null when no branch passed. */
  chosen: number | null;
  proposal?: FolderProposal<R>;
  value?: R;
  candidates: Candidate<R>[];
};

const decoder = new TextDecoder('utf-8', { fatal: true });
/** What a judgment is shown of a proposal: its result and each changed file's new text. */
function changesOf(candidate: Candidate<unknown>) {
  return { value: candidate.value, changes: candidate.proposal.diff.changes.map(change => {
    let text: string | undefined;
    try { text = change.after === undefined ? undefined : decoder.decode(change.after); } catch { text = `(${change.after!.length} bytes)`; }
    return { path: change.path, kind: change.kind, ...(text === undefined ? {} : { text }) };
  }) };
}

/**
 * Run each branch's reducer on its own private copy of `folder`, drop the branches `verify` rejects, choose one of the
 * rest (`select`, else `goal` judged pairwise, else the first) and accept it.
 */
export async function branchAndSelect<R = unknown>(folder: Workspace, branches: Branch[], options: SelectOptions<R> = {}): Promise<Selected<R>> {
  const workspace = workspaceOf(folder, 'branchAndSelect');
  if (!Array.isArray(branches) || !branches.length) throw stdError('std-arguments', 'branchAndSelect(folder, branches) needs a list of at least one branch.');
  const span = new StdSpan('branchAndSelect');
  return span.around(async () => {
    const proposals = await Promise.all(branches.map(branch => {
      const { reducer, args = [] } = typeof branch === 'object' && branch !== null && 'reducer' in branch ? branch as { reducer: Reducer; args?: unknown[] } :
        { reducer: branch };
      return span.run(() => workspace.propose<R>(reducer, ...args));
    }));
    span.stage('propose', { branches: proposals.length });
    const candidates: Candidate<R>[] = await Promise.all(proposals.map(async (proposal, index) => ({ index, proposal, value: proposal.value,
      problems: options.verify ? problemsOf(await span.run(() => options.verify!(proposal))) : [] })));
    const passing = candidates.filter(candidate => !candidate.problems.length);
    span.stage('verify', { passing: passing.map(candidate => candidate.index) });
    if (!passing.length) return { chosen: null, candidates };
    let chosen = passing[0]!;
    if (options.select) {
      const index = await span.run(() => options.select!(passing));
      const found = passing.find(candidate => candidate.index === index);
      if (!found) throw stdError('std-select-result', `select returns the index of one of the passing candidates (${passing.map(candidate => candidate.index).join(', ')}).`);
      chosen = found;
    } else if (options.goal && passing.length > 1) {
      // A knockout: each challenger meets the current choice; the earlier branch keeps ties.
      const beats = options.beats ?? shippedBeats;
      for (const challenger of passing.slice(1)) {
        const decision = await decideInFrame(span.frame, beats, [options.goal, changesOf(challenger), changesOf(chosen)]);
        if (decision.value === true) chosen = challenger;
      }
    }
    span.stage('select', { chosen: chosen.index });
    await span.run(() => workspace.accept(chosen.proposal));
    span.stage('accept', { chosen: chosen.index });
    return { chosen: chosen.index, proposal: chosen.proposal, value: chosen.value, candidates };
  }, result => result.chosen === null ? 'no branch passed' : `accepted branch ${result.chosen}`);
}
