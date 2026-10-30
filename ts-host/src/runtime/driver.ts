import type { DecisionRequest, DecisionScorer, DecisionScores, ModelTurn, ModelTurnRequest } from '../contracts.js';
import type { ModelDriver } from './runtime.js';

/**
 * A driver that wraps `inner`. `turn` answers its model turns (default: `inner`'s own) and `decide` receives the inner
 * scorer and returns the wrapped one. Every other property of `inner` is the wrapped driver's own (guidance,
 * neuralese, contextWindow, ...), so a wrapper never hides what its backend can do; the result scores replies exactly
 * where `inner` does. Every driver wrapper goes through here (contracts.ts `DecisionRequest`).
 */
export function wrapDriver(inner: ModelDriver, wrap: { turn?: ModelDriver; decide?: (inner: DecisionScorer) => DecisionScorer } = {}): ModelDriver {
  const turn = wrap.turn ?? inner;
  const wrapped: ModelDriver = Object.assign((...args: Parameters<ModelDriver>) => turn(...args), inner);
  if (inner.decide && wrap.decide) wrapped.decide = wrap.decide(inner.decide);
  return wrapped;
}

/** One request a worker thread sends its host for the host's driver. */
export type DriverCall = { kind: 'turn'; request: ModelTurnRequest } | { kind: 'decision'; request: DecisionRequest };

/**
 * The host's driver as a worker thread sees it: `call` carries each request across the thread boundary. The worker
 * scores replies exactly when the host's driver does (`decisions`), so a call's readout is the host's, not a guess.
 */
export function remoteDriver(call: (call: DriverCall) => Promise<unknown>, decisions: boolean): ModelDriver {
  const turn = (request: ModelTurnRequest) => call({ kind: 'turn', request }) as Promise<ModelTurn>;
  return decisions ? Object.assign(turn, { decide: (request: DecisionRequest) => call({ kind: 'decision', request }) as Promise<DecisionScores> }) : turn;
}
