/**
 * `refine-repair` (plans/REFINEMENT_DATA.md, plans/REFINEMENT_TYPES.md section 5.2): whole trajectories in which a
 * refinement failed and the executor repaired it.
 *
 * A teacher trajectory is the ordered turns the collector captured (`trajectoryTurn`, collector.ts): the exact context
 * each request carried and the assistant reply. When a refined `return_result` is rejected, the next turn's context ends
 * with the rejection (`rejected\nrefinement-unsatisfied: Revise the value at ... so that it is "..."`). An episode is a
 * rejection followed by a later `return_result` that stood. The record keeps every turn of the invocation up to the
 * repair, in the shape the teacher turn records have (messages / tools / target), with the loss weights the existing
 * trainers apply (`--context-weight`, `--feedback-weight`, serve/grad.py `_context_weights`):
 *
 * - instructions and inputs (everything before the first assistant reply) weigh 1, mechanical feedback after it (tool
 *   results, the rejection text) weighs `FEEDBACK_WEIGHT`, earlier assistant replies weigh 0 in later prompts because
 *   their own turn records supervise them;
 * - a rejected attempt is kept as a turn but its target is not taught as positive gold (`target_weight` 0, admission
 *   denied), like every failed proposal in the native materializer; the repair turn and the turns before the attempt are.
 *
 * Nothing here loads a model. `recordTurns` wraps a model driver to capture frames during a live run; `collectRefineRepairs`
 * turns frames (from the wrapper, or the `trajectory` of a collector row) into records.
 */
import { createHash } from 'node:crypto';
import type { ModelTurn, ModelTurnRequest } from '../contracts.js';
import { trajectoryTurn } from './collector.js';

export const REFINE_REPAIR_VERSION = 'natlang.refine-repair/1';
/** Defaults of `natlang_neuralese.train.trajectories` (`--context-weight`, `--feedback-weight`). */
export const CONTEXT_WEIGHT = 1.0;
export const FEEDBACK_WEIGHT = 0.25;

type Json = Record<string, unknown>;
type Message = { role?: string; content?: unknown; [key: string]: unknown };
export type TrajectoryFrame = Json & { phase?: string; invocation_id?: string; context?: Message[];
  assistant?: { content?: unknown; reasoning?: unknown; calls?: { tool: string; arguments: unknown }[] }; tools_offered?: unknown };
export type CollectorRow = Json & { id?: string; trajectory?: TrajectoryFrame[]; outcome?: Json; task?: Json };

const REJECTION = /^(?:rejected\n)?(refinement-unsatisfied|refinement-undecided): (.*)$/m;
const REJECTION_DETAIL = /at (\S+) so that it is ("(?:[^"\\]|\\.)*")|at (\S+) is ("(?:[^"\\]|\\.)*")/;
const sha256 = (text: string) => createHash('sha256').update(text).digest('hex');

const textOf = (content: unknown): string => typeof content === 'string' ? content
  : Array.isArray(content) ? content.map(part => typeof part === 'string' ? part : (part as { text?: unknown })?.text ?? '').join('') : '';

/** Wrap a model driver: every request/response pair is also kept as a collector frame. */
export function recordTurns<D extends (request: ModelTurnRequest) => Promise<ModelTurn>>(driver: D):
  { driver: (request: ModelTurnRequest) => Promise<ModelTurn>; frames: TrajectoryFrame[] } {
  const frames: TrajectoryFrame[] = [];
  const wrapped = Object.assign(async (request: ModelTurnRequest) => {
    const response = await driver(request);
    frames.push(trajectoryTurn(request, response) as TrajectoryFrame);
    return response;
  }, driver);
  return { driver: wrapped, frames };
}

const returnResult = (frame: TrajectoryFrame) => frame.assistant?.calls?.find(call => call.tool === 'return_result');
const isAttempt = (frame: TrajectoryFrame) => !!returnResult(frame) || !(frame.assistant?.calls?.length);

/** The rejection the executor was shown after `frame`: it sits after the last assistant message of the next context. */
function rejectionIn(context: Message[] | undefined): { code: string; text: string; message: string } | undefined {
  if (!context?.length) return undefined;
  let start = context.length;
  while (start > 0 && context[start - 1]!.role !== 'assistant') start--;
  for (const message of context.slice(start)) {
    const text = textOf(message.content);
    const match = REJECTION.exec(text);
    if (match) return { code: match[1]!, text, message: match[2]! };
  }
  return undefined;
}

function parseDetail(message: string): { path?: string; predicate?: string; probability?: number } {
  const match = REJECTION_DETAIL.exec(message);
  const path = match?.[1] ?? match?.[3], quoted = match?.[2] ?? match?.[4];
  let predicate: string | undefined;
  try { predicate = quoted ? JSON.parse(quoted) as string : undefined; } catch { predicate = undefined; }
  const probability = /probability of (\d(?:\.\d+)?)/.exec(message)?.[1];
  return { ...(path ? { path } : {}), ...(predicate === undefined ? {} : { predicate }), ...(probability === undefined ? {} : { probability: Number(probability) }) };
}

export type RefineRepairOptions = {
  /** Trace events (`refinement_check`) of the run; when given, the repair must be confirmed by a later passing check. */
  trace?: Json[];
  /** Skip rows whose outcome was not accepted (default true when the row has an outcome). */
  requireAccepted?: boolean;
};

export type RefineRepairTurn = { index: number; role: 'step' | 'rejected-attempt' | 'repair'; messages: Message[]; tools: unknown;
  target: { content: unknown; reasoning: unknown; calls: { tool: string; arguments: unknown }[] }; target_weight: number;
  training_admission: { approved: boolean; reason?: string }; raw_response_sha256: unknown };
export type RefineRepairRecord = { schema: typeof REFINE_REPAIR_VERSION; id: string; source_row: string | null; invocation_id: string | null;
  refinement: { rejections: { attempt_turn: number; feedback_turn: number; code: string; path?: string; predicate?: string; probability?: number;
    feedback: string; attempted: unknown }[]; repaired_turn: number; repaired: unknown; confirmed_by_trace: boolean | null; predicates: string[] };
  weights: { context: number; feedback: number; earlier_assistant: number; rejected_attempt_target: number };
  turns: RefineRepairTurn[] };

/**
 * Episodes of one invocation's frames, in order: a run of rejected attempts and the attempt after them that stood.
 * `base` maps a frame's position to its index in the whole trajectory. A trailing attempt with no later frame is taken
 * to have stood; `collectRefineRepairs` confirms it with the row's outcome or the trace.
 */
function episodesOf(frames: TrajectoryFrame[], base: number[]): { from: number; to: number; rejections: RefineRepairRecord['refinement']['rejections'] }[] {
  const out: { from: number; to: number; rejections: RefineRepairRecord['refinement']['rejections'] }[] = [];
  let open: { from: number; rejections: RefineRepairRecord['refinement']['rejections']; lastFeedback: number } | undefined;
  for (let k = 0; k < frames.length; k++) {
    const rejection = k > 0 && isAttempt(frames[k - 1]!) ? rejectionIn(frames[k]!.context) : undefined;
    if (rejection) {
      const previous = returnResult(frames[k - 1]!);
      const attempted = previous ? (previous.arguments as Json | undefined)?.value : frames[k - 1]!.assistant?.content;
      open ??= { from: k - 1, rejections: [], lastFeedback: k };
      open.lastFeedback = k;
      open.rejections.push({ attempt_turn: base[k - 1]!, feedback_turn: base[k]!, code: rejection.code, ...parseDetail(rejection.message),
        feedback: rejection.text, attempted });
    }
    // The attempt that stood: the first attempt at or after the last feedback that the next turn does not reject.
    if (!open || k < open.lastFeedback || !isAttempt(frames[k]!)) continue;
    const next = frames[k + 1];
    if (next && rejectionIn(next.context)) continue;
    out.push({ from: open.from, to: k, rejections: open.rejections });
    open = undefined;
  }
  return out;
}

export function collectRefineRepairs(row: CollectorRow, options: RefineRepairOptions = {}): RefineRepairRecord[] {
  const requireAccepted = options.requireAccepted ?? row.outcome !== undefined;
  if (requireAccepted && row.outcome && row.outcome.accepted !== true) return [];
  const all = (row.trajectory ?? []).map((frame, index) => ({ frame, index })).filter(({ frame }) => frame.phase === undefined || frame.phase === 'action');
  const groups = new Map<string, { frame: TrajectoryFrame; index: number }[]>();
  for (const item of all) {
    const key = item.frame.invocation_id ?? '';
    groups.set(key, [...groups.get(key) ?? [], item]);
  }
  const records: RefineRepairRecord[] = [];
  for (const [invocation, items] of groups) {
    for (const episode of episodesOf(items.map(item => item.frame), items.map(item => item.index))) {
      const slice = items.filter(item => item.index >= items[episode.from]!.index && item.index <= items[episode.to]!.index).map(item => item);
      const repairIndex = slice.at(-1)!.index;
      const rejectedAttempts = new Set(episode.rejections.map(rejection => rejection.attempt_turn));
      const predicates = [...new Set(episode.rejections.map(rejection => rejection.predicate).filter((p): p is string => !!p))];
      let confirmed: boolean | null = null;
      if (options.trace) {
        const checks = options.trace.filter(event => event.kind === 'refinement_check' && (!invocation || event.call_id === invocation || event.call_id === null || event.call_id === undefined));
        const failed = checks.some(event => event.outcome === 'fail' || event.outcome === 'undecided');
        confirmed = failed && checks.at(-1)?.outcome === 'pass';
        if (!confirmed) continue;
      }
      // Turns of the invocation before the first rejected attempt are ordinary steps of the same trajectory.
      const earlier = items.filter(item => item.index < slice[0]!.index);
      const turns: RefineRepairTurn[] = [...earlier, ...slice].map(({ frame, index }) => {
        const rejected = rejectedAttempts.has(index), repair = index === repairIndex;
        return { index, role: repair ? 'repair' as const : rejected ? 'rejected-attempt' as const : 'step' as const,
          messages: frame.context ?? [], tools: frame.tools_offered ?? [],
          target: { content: frame.assistant?.content ?? '', reasoning: frame.assistant?.reasoning ?? null, calls: frame.assistant?.calls ?? [] },
          target_weight: rejected ? 0 : 1,
          training_admission: rejected ? { approved: false, reason: 'rejected-by-refinement-check' } : { approved: true },
          raw_response_sha256: frame.raw_response_sha256 ?? null };
      });
      const repaired = returnResult(slice.at(-1)!.frame);
      const body = { source: row.id ?? null, invocation, first: slice[0]!.index, repaired: repairIndex };
      records.push({ schema: REFINE_REPAIR_VERSION, id: `refine-repair:${sha256(JSON.stringify(body)).slice(0, 20)}`, source_row: row.id ?? null,
        invocation_id: invocation || null,
        refinement: { rejections: episode.rejections, repaired_turn: repairIndex,
          repaired: repaired ? (repaired.arguments as Json | undefined)?.value : slice.at(-1)!.frame.assistant?.content, confirmed_by_trace: confirmed, predicates },
        weights: { context: CONTEXT_WEIGHT, feedback: FEEDBACK_WEIGHT, earlier_assistant: 0, rejected_attempt_target: 0 }, turns });
    }
  }
  return records;
}
