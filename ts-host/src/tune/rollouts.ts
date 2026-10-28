import { derivedSeed, sampleOption } from '../runtime/seeded-decision.js';
export { derivedSeed, sampleOption } from '../runtime/seeded-decision.js';
import { tuneRuntime, rethrowInfrastructureFailure, executionKey, type TuneExecution } from './execution.js';
/**
 * Student rollouts for program RL (plans/HIERARCHICAL_LORA_TUNING.md §5.1a).
 *
 * Each TRAINING root with a score (`expected`, or the program's verifier, tune/reward.ts) runs k times with the student serving a release, through the
 * ordinary runtime and the release's own per-call stacks. Every model turn keeps the server's exact sampling receipt
 * (`x_natlang_rl_capture`: prompt, sampled tokens with behavior log-probabilities, adapter binding, native writes and
 * imports) under its invocation identity; the runtime's execution graph (model_turn, block_write and block_read nodes)
 * binds the producer/consumer occurrences. The reward is the root score only (exact 0/1, or graded in [0, 1]). Roots without an oracle, other
 * partitions, a call that did not run its release's ready stack, a decision readout (not captured as a sampled turn)
 * and transport failures produce no rollout; they are counted in the summary with their reason.
 *
 * Output: one `natlang.tune-rollout/v1` JSON line per rollout (schema documented in natlang_neuralese/tune/rl.py),
 * consumed by the trainer's RL objective with plan field `rollouts`.
 */
import { createHash } from 'node:crypto';
import { appendFileSync, existsSync, mkdirSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import type { ProgramDescriptor } from '../adaptation/types.js';
import type { ResolvedWeightStack, WeightBinding } from '../adaptation/weights/binding.js';
import type { WeightManifest } from '../adaptation/weights/manifest.js';
import type { DecisionRequest, DecisionScores, ModelTurn, ModelTurnRequest } from '../contracts.js';
import { neuraleseServerModelTurn, type NeuraleseServerOptions } from '../model/neuralese-server.js';
import { currentFrame } from '../runtime/context.js';
import { loadNatlang } from '../runtime/node-files.js';
import { createNatlangRuntime, type ModelDriver, type Services } from '../runtime/runtime.js';
import type { SplitExample } from './collect.js';
import { exampleArguments } from './evaluate.js';
import { scorable, scoreRoot, verifierDigest, type RootScore } from './reward.js';
import { rootServices, servicesDigest } from './services.js';

export const ROLLOUT_SCHEMA = 'natlang.tune-rollout/v1';

type Json = Record<string, unknown>;
export type TuneRollout = {
  readonly schema: typeof ROLLOUT_SCHEMA; readonly root: string; readonly component: string; readonly partition: 'train';
  readonly group: string; readonly sample: number; readonly reward: RootScore;
  readonly behavior: { readonly release: string; readonly node_artifacts: Readonly<Record<string, string>>;
    readonly gates: Readonly<Record<string, number>>; readonly soft_states: Readonly<Record<string, string>>;
    readonly temperature: number; readonly seed: number };
  readonly bindings: Readonly<Record<string, string | null>>;
  readonly execution?: TuneExecution; readonly episode: Json;
};
export type RolloutSkip = { readonly root: string; readonly sample: number | null; readonly reason: string };
export type RolloutSummary = { readonly path: string; readonly written: number; readonly reused: number;
  readonly successes: number; readonly skipped: readonly RolloutSkip[] };

/** Python's `json.dumps` of a list of strings/null (the trainer's gate and soft-state keys). */
const pythonKey = (items: readonly (string | null)[]) => '[' + items.map(item => JSON.stringify(item)).join(', ') + ']';
const hash = (...parts: unknown[]) => createHash('sha256').update(JSON.stringify(parts)).digest('hex');
/** The behavior actor of a release as the trainer checks it: node versions, family gates and soft-state versions. */
export function releaseBehavior(manifest: WeightManifest) {
  return { release: manifest.digest, node_artifacts: Object.fromEntries(manifest.artifacts.map(item => [item.node, item.artifact])),
    gates: Object.fromEntries(manifest.memberships.map(item => [pythonKey([item.binding, item.family, item.version]), item.gate])),
    soft_states: Object.fromEntries(manifest.softStates.map(item => [pythonKey([item.binding, item.kind, item.target]), item.state])) };
}

/** The native reference server driver with sampling receipts requested on every turn. */
export function nativeRolloutDriver(options: NeuraleseServerOptions): ModelDriver {
  return neuraleseServerModelTurn({ ...options, request: { ...options.request, x_natlang_rl_capture: true } }) as unknown as ModelDriver;
}

class Skip extends Error {}

/**
 * Run `samples` rollouts of every oracle-bearing training root with the student `driver` serving `release`
 * (`weights` binds it, strictly). Appends to `out`; rollouts already in the file (same release, round, root and sample)
 * are reused, so a rerun continues where it stopped.
 */
export async function runRollouts(options: { root: string; program: ProgramDescriptor; release: WeightManifest;
  weights: readonly WeightBinding[]; driver: ModelDriver; examples: readonly SplitExample[]; samples: number; temperature: number;
  round: number; out: string; policyCheckpointSha256: string; log?: (line: string) => void;
  /** Host services for one root, fresh per rollout (TuneOptions.services). */
  execution?: TuneExecution; services?: (example: SplitExample) => Services }): Promise<RolloutSummary> {
  if (!Number.isInteger(options.samples) || options.samples < 2)
    throw new Error('group-relative rewards need at least two rollouts per root');
  if (!(options.temperature > 0)) throw new Error('rollouts sample the student at a positive temperature');
  mkdirSync(dirname(options.out), { recursive: true });
  const existing = new Set((existsSync(options.out) ? readFileSync(options.out, 'utf8').split('\n').filter(Boolean) : [])
    .map(line => JSON.parse(line) as TuneRollout).filter(item => item.behavior.release === options.release.digest && executionKey(item.execution) === executionKey(options.execution))
    .map(item => `${item.group}\0${item.sample}`));
  const behavior = releaseBehavior(options.release);
  const skipped: RolloutSkip[] = [];
  let written = 0, reused = 0, successes = 0;
  for (const example of options.examples) {
    if (example.partition !== 'train') { skipped.push({ root: example.id, sample: null, reason: `${example.partition} root` }); continue; }
    if (!await scorable(options.root, example)) { skipped.push({ root: example.id, sample: null, reason: 'no root oracle or verifier' }); continue; }
    const verifier = verifierDigest(options.root), services = servicesDigest(options.root);
    const group = `${example.id}/${options.release.digest}/round-${options.round}/execution-${executionKey(options.execution)}` +
      (verifier ? `/verifier-${verifier.slice(0, 16)}` : '') + (services ? `/services-${services.slice(0, 16)}` : '');
    for (let sample = 0; sample < options.samples; sample++) {
      if (existing.has(`${group}\0${sample}`)) { reused++; continue; }
      const seed = derivedSeed(options.release.digest, options.round, example.id, sample);
      try {
        const rollout = await runOne(options, example, group, sample, seed, behavior);
        appendFileSync(options.out, JSON.stringify(rollout) + '\n');
        written++; successes += rollout.reward.quality;
        options.log?.(`rollout ${example.id}#${sample}: reward ${rollout.reward.quality}`);
      } catch (error) {
        if (!(error instanceof Skip)) throw error;
        skipped.push({ root: example.id, sample, reason: error.message });
        options.log?.(`rollout ${example.id}#${sample}: skipped (${error.message})`);
      }
    }
  }
  return { path: options.out, written, reused, successes, skipped };
}

async function runOne(options: Parameters<typeof runRollouts>[0], example: SplitExample, group: string, sample: number, seed: number,
    behavior: ReturnType<typeof releaseBehavior>): Promise<TuneRollout> {
  const turns: Json[] = [], bindings: Record<string, string | null> = {}, counters = new Map<string, number>();
  const stacks: ResolvedWeightStack[] = [], events: Json[] = [];
  let transport: unknown, refused: string | undefined, fatal: Error | undefined;
  // A refusal inside the driver surfaces through the runtime as an ordinary failure; remember it so it never scores 0.
  const refuse = (reason: string): never => { refused ??= reason; throw new Skip(reason); };
  const inner = options.driver as (request: ModelTurnRequest, ...rest: unknown[]) => Promise<ModelTurn>;
  const driver = Object.assign(async (request: ModelTurnRequest, ...rest: unknown[]) => {
    const call = request.invocation_id;
    if (!call) return refuse('a model turn has no invocation identity');
    const turn = (counters.get(call) ?? 0) + 1;
    counters.set(call, turn);
    const turnSeed = derivedSeed(seed, call, turn);
    let reply: ModelTurn;
    try { reply = await inner({ ...request, temperature: options.temperature, seed: turnSeed }, ...rest); }
    catch (error) { transport ??= error; throw error; }
    const raw = reply.raw_response ?? {};
    const capture = raw.x_natlang_rl_capture as Json | undefined;
    if (!capture) throw fatal ??= new Error('rollout-capture-unavailable: the student driver must request x_natlang_rl_capture '
      + '(nativeRolloutDriver) from a native server');
    const imports = (raw.native_state as { imports?: unknown } | undefined)?.imports;
    const binding = currentFrame()?.weights?.binding ?? null;
    if (call in bindings && bindings[call] !== binding) refuse('one invocation ran under two bindings');
    bindings[call] = binding;
    const message = ((raw.choices as Json[] | undefined)?.[0]?.message ?? {}) as Json;
    turns.push({ messages: structuredClone(request.messages), ...(request.tools?.length ? { tools: structuredClone(request.tools) } : {}),
      target: { role: 'assistant', content: message.content ?? '', ...(message.tool_calls ? { tool_calls: structuredClone(message.tool_calls) } : {}) },
      invocation_id: call, sampling: { ...structuredClone(capture), ...(Array.isArray(imports) ? { native_state_imports: structuredClone(imports) } : {}) },
      replay: { raw_calls_retained: true, temperature: options.temperature, seed: turnSeed } });
    return reply;
  }, Object.fromEntries(Object.entries(options.driver))) as unknown as ModelDriver;
  // A decision readout is an action over its option set: sampled from softmax(scores / T) for exploration, with its
  // exact scoring capture and sampling receipt as the episode turn (rl/decision_policy.py).
  const decide = (options.driver as unknown as { decide?: (request: DecisionRequest, signal?: AbortSignal) => Promise<DecisionScores> }).decide;
  if (decide) (driver as unknown as { decide: unknown }).decide = async (request: DecisionRequest, signal?: AbortSignal) => {
    const call = request.invocation_id;
    if (!call) return refuse('a decision readout has no invocation identity');
    const turn = (counters.get(call) ?? 0) + 1;
    counters.set(call, turn);
    const turnSeed = derivedSeed(seed, call, turn);
    let scores: DecisionScores;
    try { scores = await decide({ ...request, native_recording: true }, signal); }
    catch (error) { transport ??= error; throw error; }
    const capture = scores.native_decision_capture;
    if (!capture) throw fatal ??= new Error('rollout-capture-unavailable: decision readouts need the native server\'s '
      + 'decision capture');
    const chosen = sampleOption(scores.log_probs, options.temperature, turnSeed);
    const binding = currentFrame()?.weights?.binding ?? null;
    if (call in bindings && bindings[call] !== binding) refuse('one invocation ran under two bindings');
    bindings[call] = binding;
    turns.push({ messages: structuredClone(request.messages), target: { role: 'assistant', content: request.options[chosen]! },
      invocation_id: call, sampling: { ...structuredClone(capture), decision_sample: { schema: 'natlang.decision-sample/1',
        log_probs: [...scores.log_probs], chosen, temperature: options.temperature, seed: turnSeed } },
      replay: { raw_calls_retained: true, temperature: options.temperature, seed: turnSeed } });
    return { ...scores, chosen };
  };
  const runtime = createNatlangRuntime({ program: options.program, ...await tuneRuntime(driver, options.execution), weights: options.weights,
    ...await rootServices(options.root, options.services, example), trace: trace => {
    const weights = (trace.adaptation as { weights?: ResolvedWeightStack } | undefined)?.weights;
    if (weights) stacks.push(weights);
    events.push(...trace.events.filter(event => typeof event.node === 'string' && Array.isArray(event.inputs)));
  } });
  let value: unknown, failed = false, executionFailure: string | undefined;
  try {
    const fn = loadNatlang(resolve(options.root, example.function), options.root);
    value = await runtime.run(() => fn(...exampleArguments(fn, example.input)));
  } catch (error) { rethrowInfrastructureFailure(error); failed = true; executionFailure = error instanceof Error ? error.message : String(error); } finally { runtime.close(); }
  if (fatal) throw fatal;
  if (refused) throw new Skip(refused);
  if (transport) throw new Skip(`model transport failed: ${transport instanceof Error ? transport.message : String(transport)}`.slice(0, 300));
  if (!turns.length) throw new Skip('the root made no model turn');
  const degraded = stacks.find(stack => stack.status !== 'ready' && stack.status !== 'untuned' || stack.effective.length < stack.requested.length);
  if (degraded) throw new Skip(`a call did not run its release stack (${degraded.status}${degraded.reason ? `: ${degraded.reason}` : ''})`);
  const reward = (await scoreRoot(options.root, example as unknown as Record<string, unknown>, value, failed))!;
  const quality = reward.quality;
  return { schema: ROLLOUT_SCHEMA, root: example.id, component: example.component, partition: 'train', group, sample,
    ...(options.execution ? { execution: options.execution } : {}), reward, behavior: { ...behavior, temperature: options.temperature, seed }, bindings,
    episode: { task_id: example.id, split: 'train', sample, group, reward: { quality },
      policy: { checkpoint_sha256: options.policyCheckpointSha256, release: options.release.digest, temperature: options.temperature },
      turns, execution_graph: events, ...(executionFailure ? { execution_failure: executionFailure.slice(0, 1000) } : {}) } };
}
