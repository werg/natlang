/**
 * Crisp structural overview of one call of a verified trajectory (DECOMPOSITION.md, part "overview"): its role,
 * prompt size, turns, each own step's turn, tool, status and size, error count, final status, and per child call its
 * function, step count, descendant count and final status. It is the cold-start view the hint writer opens with; the
 * call's contents stay in the writer's eval scope.
 */
import type { TrajectoryCall, TrajectoryOverview } from './types.js';

const size = (value: unknown): number => (typeof value === 'string' ? value : JSON.stringify(value) ?? '').length;
const descendants = (call: TrajectoryCall): number =>
  call.children.reduce((total, child) => total + 1 + descendants(child), 0);

export function trajectoryOverview(call: TrajectoryCall): TrajectoryOverview {
  return {
    call: call.call, function: call.function, role: call.role, promptChars: size(call.prompt),
    turns: new Set(call.steps.map(step => step.turn)).size,
    steps: call.steps.map((step, index) => ({ index, turn: step.turn, tool: step.tool, status: step.status,
      chars: size(step.code ?? step.arguments) + size(step.output) })),
    errors: call.steps.filter(step => step.status !== 'ok' && step.status !== 'completed').length,
    result: call.result.status,
    children: call.children.map(child => ({ call: child.call, function: child.function, steps: child.steps.length,
      descendants: descendants(child), result: child.result.status })),
  };
}
