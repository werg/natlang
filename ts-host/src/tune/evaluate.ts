import { tuneModel, type TuneExecution } from './execution.js';
/**
 * Evaluation through the ordinary engine (plans/HIERARCHICAL_LORA_TUNING.md §9). An arm is a set of weight releases
 * (none for the untuned base) run over declared root examples of one partition, with strict resolution so a missing
 * artifact fails instead of producing base results labelled as tuned. Results are per root, so arms pair by root.
 */
import { resolve } from 'node:path';
import type { ProgramDescriptor } from '../adaptation/types.js';
import type { WeightBinding, ResolvedWeightStack } from '../adaptation/weights/binding.js';
import { createNatlangRuntime, type ModelDriver, type Services } from '../runtime/runtime.js';
import { loadNatlang } from '../runtime/node-files.js';
import type { ModelTurn } from '../contracts.js';
import type { Partition, SplitExample } from './collect.js';

export type RootOutcome = {
  readonly id: string; readonly component: string; readonly partition: Partition | 'protected'; readonly function: string;
  readonly source: 'real' | 'synthetic';
  /** Whether the root has a direct oracle (`expected`); without one only completion is measured. */
  readonly oracle: boolean;
  readonly ok: boolean; readonly correct: boolean | null; readonly ms: number;
  readonly promptTokens: number; readonly completionTokens: number; readonly turns: number;
  /** Every call's recorded stack: identity, status and how many adapters it really ran. */
  readonly stacks: readonly { readonly identity: string; readonly status: string; readonly requested: number; readonly effective: number }[];
  readonly error?: string;
};
export type ArmResult = { readonly schema: 'natlang.tune-arm/v1'; readonly arm: string; readonly release: string | null;
  readonly partition: string; readonly execution?: TuneExecution; readonly roots: readonly RootOutcome[] };

/** The callable's positional arguments for an example input (an object is keyed by parameter name). */
export function exampleArguments(fn: unknown, input: unknown): unknown[] {
  const params = (fn as Record<symbol, { definition: { params: { name: string }[] } }>)[Symbol.for('natlang.callable')]!.definition.params;
  return input && typeof input === 'object' && !Array.isArray(input) ? params.map(p => (input as Record<string, unknown>)[p.name])
    : Array.isArray(input) ? input : [input];
}

const equal = (left: unknown, right: unknown) => JSON.stringify(left) === JSON.stringify(right);

/** Run one arm over the examples, one fresh task per root. */
export async function evaluateRoots(options: { root: string; program: ProgramDescriptor; driver: ModelDriver;
  weights: readonly WeightBinding[]; examples: readonly SplitExample[]; arm: string; release: string | null; partition: string;
  /** Host services for one root, fresh per root (TuneOptions.services). */
  execution?: TuneExecution; services?: (example: SplitExample) => Services }): Promise<ArmResult> {
  const roots: RootOutcome[] = [];
  for (const example of options.examples) {
    const stacks: RootOutcome['stacks'][number][] = [];
    let promptTokens = 0, completionTokens = 0, turns = 0;
    const inner = options.driver as (...args: unknown[]) => Promise<ModelTurn>;
    // Same driver capabilities (decide, contextWindow, Neuralese marker), counting the turns it serves.
    const driver = Object.assign(async (...args: unknown[]) => {
      const turn = await inner(...args);
      turns++; promptTokens += turn.prompt_tokens ?? 0; completionTokens += turn.completion_tokens ?? 0;
      return turn;
    }, Object.fromEntries(Object.entries(options.driver))) as unknown as ModelDriver;
    const runtime = createNatlangRuntime({ program: options.program, model: tuneModel(driver, options.execution), weights: options.weights,
      ...(options.services ? { services: options.services(example) } : {}), trace: trace => {
      const recorded = (trace.adaptation as { weights?: ResolvedWeightStack } | undefined)?.weights;
      if (recorded) stacks.push({ identity: recorded.identity, status: recorded.status, requested: recorded.requested.length, effective: recorded.effective.length });
    } });
    const started = Date.now();
    let ok = false, correct: boolean | null = null, error: string | undefined;
    try {
      const fn = loadNatlang(resolve(options.root, example.function), options.root);
      const value = await runtime.run(() => fn(...exampleArguments(fn, example.input)));
      ok = true;
      if ('expected' in example) correct = equal(value, example.expected);
    } catch (failure) {
      error = failure instanceof Error ? failure.message : String(failure);
      if ('expected' in example) correct = false;
    } finally { runtime.close(); }
    roots.push({ id: example.id, component: example.component, partition: example.protected ? 'protected' : example.partition,
      function: example.function, source: example.source ?? 'real', oracle: 'expected' in example, ok, correct, ms: Date.now() - started,
      promptTokens, completionTokens, turns, stacks, ...(error ? { error: error.slice(0, 500) } : {}) });
  }
  return { schema: 'natlang.tune-arm/v1', arm: options.arm, release: options.release, partition: options.partition, ...(options.execution ? { execution: options.execution } : {}), roots };
}

/** Success of one root: the oracle when it has one, completion otherwise. */
export const rootSuccess = (outcome: RootOutcome): number => (outcome.oracle ? outcome.correct === true : outcome.ok) ? 1 : 0;

export function armSummary(arm: ArmResult) {
  const roots = arm.roots, n = roots.length || 1;
  const calls = roots.flatMap(root => root.stacks);
  return { arm: arm.arm, release: arm.release, roots: roots.length, success: roots.reduce((sum, root) => sum + rootSuccess(root), 0) / n,
    completed: roots.filter(root => root.ok).length, meanMs: roots.reduce((sum, root) => sum + root.ms, 0) / n,
    tokens: roots.reduce((sum, root) => sum + root.promptTokens + root.completionTokens, 0),
    calls: calls.length, degradedCalls: calls.filter(call => call.status === 'degraded').length,
    fallbackCalls: calls.filter(call => call.requested > 0 && call.effective < call.requested).length,
    distinctStacks: new Set(calls.filter(call => call.effective > 0).map(call => call.identity)).size };
}
