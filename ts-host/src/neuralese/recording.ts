/**
 * Recording model turns for gradient replay (spec/NEURALESE_GRAPH.md, "Replay"; `natlang:learning`).
 *
 * A Neuralese-capable driver reports every exchange it makes to the active recorder: the request it sent (messages
 * with block parts, tools) and the assistant message the server returned (with block parts and the write records of
 * the blocks it wrote). The learning module installs the source of the active recorder (Node: `AsyncLocalStorage`),
 * so this module stays platform-neutral.
 */
export type RecordedTurn = {
  readonly messages: unknown[];
  readonly tools?: unknown[];
  /** The assistant message as the server returned it: content and tool-call arguments may hold block parts. */
  readonly reply: Record<string, unknown>;
  /** Write records of the blocks this turn wrote. */
  readonly blocks: readonly Record<string, unknown>[];
  /** A decision readout instead of a generated reply: the options it scored (native/decision.ts). */
  readonly decision?: { readonly options: readonly string[] };
};

export interface TurnRecorder { record(turn: RecordedTurn): void }

let source: () => TurnRecorder | undefined = () => undefined;

/** Install where the active recorder comes from (the learning module does this). */
export function setRecorderSource(next: () => TurnRecorder | undefined): void { source = next; }

/** The recorder of the current asynchronous context, if any. */
export function activeRecorder(): TurnRecorder | undefined { return source(); }
