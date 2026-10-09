/**
 * Fused hand-offs at run time (plans/FUSED_PIPELINES.md). A fused edge is a planned hand-off from a producer function
 * to a consumer function inside one orchestrator: the value stays a Neuralese block instead of being decoded to text and
 * written again. Authors never see it: the declared types stay `T`. At a call the kernel asks `engageFusion`, which for
 * a planned edge changes only the call's effective signature, the producer's result to `Neuralese<T>` and the
 * consumer's parameter to `Neuralese<T>`. Everything else is the existing soft-value machinery (typed result writes,
 * soft arguments, the store and the port).
 *
 * Modes: `off` (default) changes nothing and traces nothing; `on` fuses planned edges when the model profile carries a
 * runtime-qualification certificate for these exact weights and dialect, and otherwise falls back to text and traces
 * `fusion_fallback` with the reason; `shadow` serves text and measures the fused producer beside it (`fusion_shadow`).
 *
 * Trace events (on the orchestrator's call): `fusion_edge` (status `fused` | `consumed` | `fallback`), `fusion_fallback`,
 * `fusion_shadow`. A run on the text emulation (model/text-neuralese-emulation.ts) stands in for a qualified channel
 * only where the host says so with `emulation`; every event then carries `emulation: true`.
 */
import { traceFor } from '../native/graph.js';
import { markFusedBlock, setCurrentTaskResolver } from '../native/fusion-blocks.js';
import { isNeuraleseRef, supportsNeuralese, type NeuraleseRef, type NeuraleseRuntimeOptions } from '../native/neuralese.js';
import { currentFrame, type Frame } from './context.js';

setCurrentTaskResolver(() => currentFrame()?.task);

export type FusionMode = 'off' | 'shadow' | 'on';
export const FUSION_CERTIFICATE_SCHEMA = 'natlang.fusion-certificate/1';

/** A fused edge, as the runtime needs it. `src/fusion` derives these from the facts and the verified plan. */
export type FusionEdgeSpec = {
  id: string;
  /** Source of the orchestrating function whose call made both calls. */
  scope: string;
  producer: { source: string };
  consumer: { source: string; param: string };
  /** The declared result type of the producer (`T`). */
  type: string;
  /**
   * Edges of crisp TypeScript orchestrators have no calling function to read the scope from. The compiler marks the two
   * calls instead (`__natlang.fuseSite`), and these are their site IDs (`file@offset`, compiler/call-flow.ts).
   */
  sites?: { producer: string; consumer: string };
};

/**
 * What the training pipeline issues once the exact weights pass runtime qualification (TRAINING_RECIPE.md stages 4-5):
 * transport, gradient replay and typed function execution on the production encode/read/write path, and the
 * self-feedback channel-equivalence and generation-quality gates (DECISIONS.md, 2026-10-09).
 */
export type FusionCertificate = {
  schema: typeof FUSION_CERTIFICATE_SCHEMA;
  model: { id: string; weights_sha256: string };
  /** The dialect tag the weights read and write (`nd:<name>@<version>`). */
  dialect: string;
  runtime_qualification: { status: 'passed' | 'failed'; scope: string[]; report_sha256: string;
    self_feedback?: { channel_equivalence: 'passed' | 'failed'; generation_quality: 'passed' | 'failed' } };
  foundation_certificate_sha256?: string;
  issued_at?: string; issuer?: string;
};

export type FusionOptions = {
  mode: FusionMode;
  edges: readonly FusionEdgeSpec[];
  certificate?: FusionCertificate;
  /** Digest of the weights the active profile serves; the certificate must name the same. */
  weights?: string;
  /** The runtime's Neuralese channel is the text emulation with this dialect; it stands in for a certificate. */
  emulation?: { dialect: string };
  /** Decode a fused block back to a value, for shadow comparison (default: the typed `read`). */
  readback?: (ref: NeuraleseRef) => Promise<unknown>;
  /** Whether two values agree (default: equal JSON). */
  agree?: (text: unknown, fused: unknown) => boolean;
};

export type FusionStatus = { ok: true; emulation: boolean } | { ok: false; reason: string };

/** Whether fusion may engage for a call on `model`: the backend carries Neuralese and the certificate matches. */
export function fusionStatus(options: FusionOptions, neuralese: NeuraleseRuntimeOptions | undefined,
    model: { id?: string; driver: unknown } | undefined): FusionStatus {
  if (!neuralese?.store || !neuralese.port) return { ok: false, reason: 'the runtime has no Neuralese store and write port (backend lacks Neuralese capability)' };
  if (options.emulation) {
    if (neuralese.port.dialect !== options.emulation.dialect)
      return { ok: false, reason: `the emulation is declared for ${options.emulation.dialect}, the port writes ${neuralese.port.dialect}` };
    return { ok: true, emulation: true };
  }
  if (!model || !supportsNeuralese(model.driver)) return { ok: false, reason: 'the model backend does not carry Neuralese content (neuralese-unsupported-backend)' };
  const certificate = options.certificate;
  if (!certificate) return { ok: false, reason: `no runtime-qualification certificate for model ${model.id ?? 'unknown'}` };
  if (certificate.schema !== FUSION_CERTIFICATE_SCHEMA) return { ok: false, reason: `certificate schema ${String(certificate.schema)} is not ${FUSION_CERTIFICATE_SCHEMA}` };
  if (certificate.runtime_qualification?.status !== 'passed') return { ok: false, reason: 'the certificate does not record a passed runtime qualification' };
  const feedback = certificate.runtime_qualification.self_feedback;
  if (!feedback || feedback.channel_equivalence !== 'passed' || feedback.generation_quality !== 'passed')
    return { ok: false, reason: 'the certificate does not record the passed self-feedback gates (channel equivalence and generation quality)' };
  if (!model.id || certificate.model?.id !== model.id) return { ok: false, reason: `the certificate is for model ${certificate.model?.id ?? 'unknown'}, the profile serves ${model.id ?? 'an unnamed model'}` };
  if (!options.weights || certificate.model.weights_sha256 !== options.weights)
    return { ok: false, reason: 'the certificate names different weights than the profile serves' };
  if (certificate.dialect !== neuralese.port.dialect)
    return { ok: false, reason: `the certificate is for dialect ${certificate.dialect}, the port writes ${neuralese.port.dialect}` };
  return { ok: true, emulation: false };
}

const sameSource = (a: string, b: string): boolean => a === b || a.endsWith(`/${b}`) || b.endsWith(`/${a}`);
/** `file@offset` IDs are equal when the offsets are and the files are the same file up to a leading directory. */
const sameSite = (a: string, b: string): boolean => {
  const at = a.lastIndexOf('@'), other = b.lastIndexOf('@');
  return at > 0 && other > 0 && a.slice(at) === b.slice(other) && sameSource(a.slice(0, at), b.slice(0, other));
};
const canonical = (value: unknown): string => JSON.stringify(value, (_key, item) => item && typeof item === 'object' && !Array.isArray(item) ?
  Object.fromEntries(Object.entries(item).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)) : item);

/** What the kernel asks of a call it is about to run. */
export type FusionEngagement = {
  /** The definition to run: its result or one parameter retyped to `Neuralese<T>` on a fused edge; else unchanged. */
  definition: FusedDefinition;
  /** Called with the call's host value once it has one (shadow: starts the comparison). */
  produced?: (value: unknown) => void;
};
type FusedDefinition = { returns: string; params: { name: string; type: string; optional?: boolean }[] };

type EdgeState = { fallbackReported: boolean; fused: number; consumed: number; shadows: number };
const STATES = new WeakMap<object, Map<string, EdgeState>>();
const HOST_EVENTS = new WeakMap<object, { kind: string; data: Record<string, unknown> }[]>();

/** Fusion events of calls the host made (crisp TypeScript orchestrators), which have no calling function's trace to join. */
export function fusionHostEvents(task: object): readonly { kind: string; data: Record<string, unknown> }[] { return HOST_EVENTS.get(task) ?? []; }
const UNCONSUMED = new WeakMap<object, Set<string>>();

/** Edges that produced a fused block no consumer took, per task (a planned consumer never received it). */
export function unconsumedFusedBlocks(task: object): string[] { return [...(UNCONSUMED.get(task) ?? [])]; }

function emit(frame: Frame, kind: string, data: Record<string, unknown>): void {
  const trace = traceFor(frame.parentCallId);
  if (trace) { trace.emit(kind, { call_id: frame.parentCallId ?? null, ...data }); return; }
  const events = HOST_EVENTS.get(frame.task) ?? (HOST_EVENTS.set(frame.task, []), HOST_EVENTS.get(frame.task)!);
  events.push({ kind, data });
}

/**
 * Decide, for the call of `definition` with `inputs` made inside `frame`, whether a planned edge applies, and retype
 * the call if so. `invoke` runs a call the way the kernel does (used by shadow measurement). Returns nothing when
 * fusion is off or no planned edge involves this call.
 */
export function engageFusion<D extends FusedDefinition & { source?: string; name: string; types?: Record<string, string> }>(
    frame: Frame, definition: D, inputs: readonly unknown[],
    invoke: (frame: Frame, definition: D, inputs: unknown[]) => Promise<unknown>): { definition: D; produced?: (value: unknown) => void } | undefined {
  const options = frame.task.runtime.options.fusion;
  if (!options || options.mode === 'off' || !definition.source || (definition as { subtype?: string }).subtype === 'directory-reducer') return undefined;
  const soft = (type: string) => `Neuralese<${type}>`;
  let asProducer: FusionEdgeSpec[], asConsumer: FusionEdgeSpec[];
  if (frame.fusionSite) {
    // A call the compiler marked in crisp TypeScript: the edge names the call sites, and the function must be the edge's own.
    const site = frame.fusionSite;
    asProducer = /\bNeuralese\s*</.test(definition.returns) ? [] : options.edges.filter(edge => edge.sites && sameSite(edge.sites.producer, site) && sameSource(edge.producer.source, definition.source!));
    asConsumer = options.edges.filter(edge => edge.sites && sameSite(edge.sites.consumer, site) && sameSource(edge.consumer.source, definition.source!));
  } else {
    if (!frame.parentCallId) return undefined;
    const parent = (traceFor(frame.parentCallId)?.events[0] ?? {}) as Record<string, unknown>;
    const scope = typeof parent.definition_source === 'string' ? parent.definition_source : undefined;
    if (!scope) return undefined;
    asProducer = /\bNeuralese\s*</.test(definition.returns) ? [] : options.edges.filter(edge => !edge.sites && sameSource(edge.scope, scope) && sameSource(edge.producer.source, definition.source!));
    asConsumer = options.edges.filter(edge => !edge.sites && sameSource(edge.scope, scope) && sameSource(edge.consumer.source, definition.source!));
  }
  if (!asProducer.length && !asConsumer.length) return undefined;
  const states = STATES.get(frame.task) ?? (STATES.set(frame.task, new Map()), STATES.get(frame.task)!);
  const state = (edge: FusionEdgeSpec): EdgeState => states.get(edge.id) ?? (states.set(edge.id, { fallbackReported: false, fused: 0, consumed: 0, shadows: 0 }), states.get(edge.id)!);
  const model = frame.task.model((definition as { model?: string }).model);
  const status = fusionStatus(options, frame.task.runtime.options.neuralese, model);
  const mark = status.ok && status.emulation ? { emulation: true } : {};
  let result: D = definition;

  // The consumer receives a block a fused producer wrote: retype the parameter. A text value (the producer fell back or
  // ran in shadow) leaves the call as written.
  if (options.mode === 'on' && status.ok) for (const edge of asConsumer) {
    const index = definition.params.findIndex(param => param.name === edge.consumer.param);
    const value = index >= 0 ? inputs[index] : undefined;
    if (index < 0 || !isNeuraleseRef(value)) continue;
    result = { ...result, params: result.params.map((param, at) => at === index ? { ...param, type: soft(param.type) } : param) };
    const edgeState = state(edge);
    edgeState.consumed++;
    UNCONSUMED.get(frame.task)?.delete(value.$neuralese.id);
    emit(frame, 'fusion_edge', { edge: edge.id, status: 'consumed', producer: edge.producer.source, consumer: edge.consumer.source,
      param: edge.consumer.param, block: value.$neuralese.id, ...mark });
  }

  const edge = asProducer[0];
  if (!edge) return result === definition ? undefined : { definition: result };
  if (!status.ok) {
    const edgeState = state(edge);
    if (!edgeState.fallbackReported) {
      edgeState.fallbackReported = true;
      emit(frame, 'fusion_fallback', { edge: edge.id, mode: options.mode, reason: status.reason, producer: edge.producer.source, consumer: edge.consumer.source });
      emit(frame, 'fusion_edge', { edge: edge.id, status: 'fallback', reason: status.reason });
    }
    return result === definition ? undefined : { definition: result };
  }
  if (options.mode === 'on') {
    state(edge).fused++;
    emit(frame, 'fusion_edge', { edge: edge.id, status: 'fused', producer: edge.producer.source, consumer: edge.consumer.source, type: edge.type, ...mark });
    result = { ...result, returns: soft(definition.returns) };
    return { definition: result, produced: value => {
      if (isNeuraleseRef(value)) {
        const pending = UNCONSUMED.get(frame.task) ?? (UNCONSUMED.set(frame.task, new Set()), UNCONSUMED.get(frame.task)!);
        pending.add(value.$neuralese.id);
        markFusedBlock(frame.task, value.$neuralese.id);
      }
    } };
  }
  // Shadow: the text call is served. Beside it, run the same producer with a soft result, read the block back and compare.
  return { definition: result, produced: textValue => {
    state(edge).shadows++;
    const call = (async () => {
      const started = Date.now();
      const shadowDefinition: D = { ...definition, returns: soft(definition.returns) };
      let fused: unknown, error: string | undefined;
      try { fused = await invoke(frame, shadowDefinition, [...inputs]); } catch (caught) { error = caught instanceof Error ? caught.message : String(caught); }
      let decoded: unknown, agree: boolean | null = null;
      if (!error && isNeuraleseRef(fused)) {
        try {
          const read = options.readback ?? (await import('../neuralese/combinators.js')).readNeuraleseForCurrentTask;
          decoded = await read(fused);
          agree = (options.agree ?? ((a, b) => canonical(a) === canonical(b)))(textValue, decoded);
        } catch (caught) { error = `readback failed: ${caught instanceof Error ? caught.message : String(caught)}`; }
      } else if (!error) error = 'the shadow producer did not return a Neuralese block';
      emit(frame, 'fusion_shadow', { edge: edge.id, producer: edge.producer.source, consumer: edge.consumer.source, agree,
        ...(error ? { error } : {}), ...(fused && isNeuraleseRef(fused) ? { block: fused.$neuralese.id } : {}),
        shadow_ms: Date.now() - started, scope: 'producer', ...mark });
    })();
    call.catch(() => {});
    frame.task.track(call, frame.parentCallId);
  } };
}
