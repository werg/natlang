import type { DecisionRequest, ModelTurnOptions, ModelTurnRequest } from '../contracts.js';
import type { ModelDriver } from '../runtime/runtime.js';
import type { ManagedModelSession } from './local-server.js';

/** A model session's turns, decisions and context window as a runtime model driver. */
export function sessionDriver(session: ManagedModelSession): ModelDriver {
  return Object.assign((request: ModelTurnRequest, signal?: AbortSignal, options?: ModelTurnOptions) => session.turn(request, signal, options),
    { decide: (request: DecisionRequest, signal?: AbortSignal) => session.decide(request, signal), contextWindow: () => session.contextWindow(),
      ...(session.guidance ? { guidance: true as const } : {}),
      ...(session.neuralese ? { neuralese: true as const, neuraleseRuntime: () => session.neuraleseRuntime!() } : {}) });
}
