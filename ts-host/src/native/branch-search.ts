/**
 * Prefix-shared `branch@N` search & Step-Verifier ranking (`plans/NATLANG_COMPILER.md` §6.2, Track 4).
 *
 * Fans out N continuations sharing the exact prompt prefix (near-zero extra prefill cost on SGLang / llama.cpp),
 * ranks branches by crisp checks (`!malformed`, `!truncated`, valid `return_result` schema / `ensures`) strictly
 * before the `StepVerifierHead` score `V(h_t)`, and emits a `BranchWinRecord` whenever branch@N rescues a turn
 * that branch 0 (pass@1) would have failed — closing the self-play flywheel into OPSD, T2, and `ScopedExperienceKB`.
 */
import type { ModelTurn, ModelTurnRequest } from '../contracts.js';

export interface BranchCandidateEvaluation {
  index: number;
  turn: ModelTurn;
  crispOk: boolean;
  crispReason?: string;
  verifierScore: number;
}

export interface BranchWinRecord {
  callId: string | null;
  turnNumber: number;
  branches: number;
  selectedIndex: number;
  pass1CrispOk: boolean;
  selectedCrispOk: boolean;
  pass1Score: number;
  selectedScore: number;
  selectedTurn: ModelTurn;
}

export interface BranchSearchOptions {
  /** Number of prefix-shared candidate continuations to sample (typically 4). */
  branches: number;
  /** Sampling temperature for branches 1..N-1 when base temperature is 0 (default 0.6). */
  branchTemperature?: number;
  /** Optional StepVerifierHead scorer returning V(h_t) in [0, 1] for each branch. */
  scoreBranch?: (turn: ModelTurn, index: number, request: ModelTurnRequest) => Promise<number> | number;
  /** Optional crisp validator for a proposed return_result value (e.g. hoisted contract + ensures). */
  validateReturnValue?: (value: unknown) => Promise<{ ok: boolean; reason?: string }> | { ok: boolean; reason?: string };
  /** Callback invoked whenever branch@N beats branch 0 (self-play win harvesting). */
  onBranchWin?: (win: BranchWinRecord) => void;
}

/** Deterministic crisp-first, verifier-second branch ranking (mirrors `step_verifier.py::rank_branches`). */
export function rankBranchCandidates(candidates: readonly BranchCandidateEvaluation[]): BranchCandidateEvaluation[] {
  return [...candidates].sort((a, b) => {
    if (a.crispOk !== b.crispOk) return a.crispOk ? -1 : 1;
    if (b.verifierScore !== a.verifierScore) return b.verifierScore - a.verifierScore;
    return a.index - b.index;
  });
}

async function evaluateCrisp(
  turn: ModelTurn,
  validateReturnValue?: BranchSearchOptions['validateReturnValue'],
): Promise<{ ok: boolean; reason?: string }> {
  if (turn.malformed_call) return { ok: false, reason: `malformed_call:${turn.malformed_call}` };
  if (turn.protocol_failure) return { ok: false, reason: 'protocol_failure' };
  if (turn.truncated && !(turn.calls?.length)) return { ok: false, reason: 'truncated_without_call' };
  const calls = turn.calls ?? [];
  if (validateReturnValue) {
    for (const [name, args] of calls) {
      if (name === 'return_result') {
        if (args && args.status === 'success' && Object.hasOwn(args, 'value')) {
          const verdict = await validateReturnValue(args.value);
          if (!verdict.ok) return { ok: false, reason: verdict.reason ?? 'return_contract_failed' };
        }
      }
    }
  }
  return { ok: true };
}

/** Execute N prefix-shared continuations in parallel and select the highest-ranked branch. */
export async function runPrefixSharedBranchSearch(
  baseRequest: ModelTurnRequest,
  sampleTurn: (request: ModelTurnRequest, branchIndex: number) => Promise<ModelTurn> | ModelTurn,
  options: BranchSearchOptions,
  meta: { callId: string | null; turnNumber: number },
): Promise<{ selected: BranchCandidateEvaluation; ranked: BranchCandidateEvaluation[]; branchWin: boolean }> {
  const n = Math.max(1, Math.floor(options.branches));
  const baseSeed = typeof baseRequest.seed === 'number' ? baseRequest.seed : 0;
  const branchTemp = options.branchTemperature ?? 0.6;

  // Branch 0 is the exact pass@1 request, unchanged (an unset seed stays unset); branches 1..N-1 sample from the shared
  // prefix at the request's own temperature (or `branchTemperature` when it is greedy) with distinct seeds when it has one.
  const requests: ModelTurnRequest[] = Array.from({ length: n }, (_, index) => index === 0 ? baseRequest : ({
    ...baseRequest,
    temperature: baseRequest.temperature && baseRequest.temperature > 0 ? baseRequest.temperature : branchTemp,
    ...(typeof baseRequest.seed === 'number' ? { seed: baseSeed + index * 1009 } : {}),
  }));

  const turns = await Promise.all(requests.map((req, index) => sampleTurn(req, index)));
  const evaluated: BranchCandidateEvaluation[] = await Promise.all(
    turns.map(async (turn, index) => {
      const crisp = await evaluateCrisp(turn, options.validateReturnValue);
      let score = 0.5;
      if (options.scoreBranch) {
        const raw = await options.scoreBranch(turn, index, requests[index]!);
        score = Number.isFinite(raw) ? raw : 0.5;
      }
      return { index, turn, crispOk: crisp.ok, ...(crisp.reason ? { crispReason: crisp.reason } : {}), verifierScore: score };
    }),
  );

  const ranked = rankBranchCandidates(evaluated);
  const selected = ranked[0]!;
  const pass1 = evaluated[0]!;
  const branchWin = selected.index !== 0 && (
    (!pass1.crispOk && selected.crispOk) ||
    (pass1.crispOk === selected.crispOk && selected.verifierScore > pass1.verifierScore + 1e-6)
  );
  if (branchWin && options.onBranchWin) {
    options.onBranchWin({
      callId: meta.callId,
      turnNumber: meta.turnNumber,
      branches: n,
      selectedIndex: selected.index,
      pass1CrispOk: pass1.crispOk,
      selectedCrispOk: selected.crispOk,
      pass1Score: pass1.verifierScore,
      selectedScore: selected.verifierScore,
      selectedTurn: selected.turn,
    });
  }
  return { selected, ranked, branchWin };
}
