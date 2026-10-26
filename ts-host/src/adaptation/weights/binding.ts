/**
 * Per-call weight-stack resolution (plans/HIERARCHICAL_LORA_TUNING.md §6). A `WeightBinding` is one immutable
 * release of one program, pinned by a task at creation; existing tasks finish on it while new tasks resolve the next.
 *
 * At a call boundary the callee's binding gives its stack: gated families, then its scope ancestry root to leaf.
 * That stack replaces the caller's lexical stack; explicit `withNativeAdapters` scopes append after it (the model
 * driver combines them). Requested and effective stacks are both recorded. An artifact the backend cannot serve is
 * dropped together with every artifact that requires it; the reduced stack runs only if it was validated, otherwise
 * the declared base fallback runs and the call is marked degraded, never reported as tuned execution. Strict
 * resolution (evaluation, export validation) throws instead.
 */
import { fingerprint } from '../identity.js';
import { nativeModelIdentityEqual, type NativeModelIdentity } from '../../native/neuralese-store.js';
import type { NativeAdapterBinding } from '../../native/model-recording.js';
import type { ProgramDescriptor } from '../types.js';
import { bindingSemanticHash } from './scopes.js';
import type { WeightBindingEntry, WeightManifest } from './manifest.js';

export type WeightStatus = 'ready' | 'stale' | 'degraded';
export type StackEntry = NativeAdapterBinding & { readonly node: string };
/** What one call runs with, as recorded in its trace. */
export type ResolvedWeightStack = {
  readonly release: string; readonly program: string; readonly binding: string | null;
  readonly requested: readonly StackEntry[]; readonly effective: readonly StackEntry[];
  readonly skipped: readonly { readonly artifact: string; readonly node: string; readonly reason: string }[];
  readonly status: WeightStatus | 'untuned'; readonly reason?: string;
  /** Directly tuned instruction/skill states this call consumes in place of their exact texts (plan §3.3). */
  readonly soft?: { readonly instructions?: SoftUse; readonly skills: Readonly<Record<string, SoftUse>> };
  /** Canonical identity of the effective stack over the exact base; equal stacks share it. */
  readonly identity: string;
  /** The identity of the untuned base executor for this release (what a fallback runs). */
  readonly baseIdentity: string;
};

export type SoftUse = { readonly state: string; readonly textSha256: string };

export class WeightBindingError extends Error {
  constructor(readonly code: 'weights-incompatible' | 'weights-unavailable' | 'weights-stale', message: string) {
    super(`${code}: ${message}`); this.name = 'WeightBindingError';
  }
}

export type WeightBindingOptions = {
  /** The executing base. A release trained on another base is incompatible and falls back as a whole. */
  readonly executor?: NativeModelIdentity;
  /** Artifacts the selected backend acknowledged as loaded; absent means all were acknowledged at activation. */
  readonly available?: ReadonlySet<string>;
  /** The program as currently compiled, to detect bindings whose semantics changed after training. */
  readonly program?: ProgramDescriptor;
  /** Evaluation/export validation: missing, stale or incompatible artifacts throw instead of falling back. */
  readonly strict?: boolean;
  /** The serving backend found the release incompatible (for example its GGUF exports bind another base file). */
  readonly incompatible?: string;
  /** Whether the serving backend imports stored native state; soft instruction/skill states need it. */
  readonly nativeState?: boolean;
};

export function stackIdentity(base: NativeModelIdentity, stack: readonly StackEntry[], soft?: ResolvedWeightStack['soft']): string {
  return fingerprint({ base, stack: stack.map(entry => ({ id: entry.id, scale_hex: scaleHex(entry.scale) })), ...(soft ? { soft } : {}) },
    'natlang.weights-stack/v1');
}

/** The same call run untuned (a soft text no longer matches its trained text): base executor, marked degraded. */
export function untunedStack(stack: ResolvedWeightStack, reason: string): ResolvedWeightStack {
  const { soft: _soft, ...rest } = stack;
  return { ...rest, effective: [], status: 'degraded', reason, identity: stack.baseIdentity };
}

/** Exact float text, matching Python's float.hex() for the identity hash. */
function scaleHex(value: number): string {
  if (value === 0) return Object.is(value, -0) ? '-0x0.0p+0' : '0x0.0p+0';
  const view = new DataView(new ArrayBuffer(8)); view.setFloat64(0, value);
  const high = view.getUint32(0), low = view.getUint32(4);
  const sign = high >>> 31 ? '-' : '', exponent = (high >>> 20) & 0x7ff;
  const mantissa = ((BigInt(high & 0xfffff) << 32n) | BigInt(low)).toString(16).padStart(13, '0');
  return exponent === 0 ? `${sign}0x0.${mantissa}p-1022` : `${sign}0x1.${mantissa}p${exponent - 1023 >= 0 ? '+' : ''}${exponent - 1023}`;
}

export class WeightBinding {
  private readonly byDefinition = new Map<string, WeightBindingEntry>();
  private readonly byComponent = new Map<string, WeightBindingEntry>();
  private readonly cache = new Map<string, ResolvedWeightStack>();
  readonly status: WeightStatus;
  readonly reason?: string;
  private readonly staleBindings = new Set<string>();

  constructor(readonly manifest: WeightManifest, readonly options: WeightBindingOptions = {}) {
    for (const entry of manifest.bindings) {
      const key = (id: string) => JSON.stringify([id, entry.contextId]);
      if (entry.definitionId) this.byDefinition.set(key(entry.definitionId), entry);
      this.byComponent.set(key(entry.component), entry);
    }
    let status: WeightStatus = 'ready', reason: string | undefined;
    if (options.incompatible) {
      status = 'degraded'; reason = options.incompatible;
      if (options.strict) throw new WeightBindingError('weights-incompatible', options.incompatible);
    } else if (options.executor && !nativeModelIdentityEqual(options.executor, manifest.base)) {
      status = 'degraded'; reason = 'the release was trained on another base; it runs the base model until retuned';
      if (options.strict) throw new WeightBindingError('weights-incompatible', `release ${manifest.digest.slice(0, 12)} was trained on another exact base`);
    } else if (options.program) {
      const components = new Map(options.program.components.map(component => [component.key, component]));
      for (const entry of manifest.bindings) {
        const component = components.get(entry.component);
        if (!component || bindingSemanticHash(component) !== entry.semanticHash) this.staleBindings.add(entry.id);
      }
      if (this.staleBindings.size || options.program.buildHash !== manifest.program.buildHash) {
        status = 'stale'; reason = this.staleBindings.size ? `${this.staleBindings.size} definition(s) changed since tuning; they run untuned until \`natlang tune --incremental\``
          : 'program sources changed since tuning; run `natlang tune --incremental` to refresh';
        if (options.strict && this.staleBindings.size) throw new WeightBindingError('weights-stale', reason);
      }
    }
    this.status = status; this.reason = reason;
  }

  get release(): string { return this.manifest.digest; }
  get programId(): string { return this.manifest.program.id; }

  /** The stack for one call of a definition (by definition ID or component key) in a context (null: file context). */
  resolve(definitionId: string, component: string | undefined, contextId: string | null): ResolvedWeightStack {
    const cacheKey = JSON.stringify([definitionId, component ?? null, contextId]);
    const cached = this.cache.get(cacheKey);
    if (cached) return cached;
    const key = (id: string) => JSON.stringify([id, contextId]);
    const entry = this.byDefinition.get(key(definitionId)) ?? (component ? this.byComponent.get(key(component)) : undefined);
    const resolved = this.resolveEntry(entry, contextId);
    this.cache.set(cacheKey, resolved);
    return resolved;
  }

  private base(binding: string | null, requested: readonly StackEntry[], skipped: ResolvedWeightStack['skipped'],
    status: ResolvedWeightStack['status'], reason?: string): ResolvedWeightStack {
    return { release: this.release, program: this.programId, binding, requested, effective: [], skipped, status,
      ...(reason ? { reason } : {}), identity: stackIdentity(this.manifest.base, []), baseIdentity: stackIdentity(this.manifest.base, []) };
  }

  private resolveEntry(entry: WeightBindingEntry | undefined, contextId: string | null): ResolvedWeightStack {
    if (!entry) {
      // A rebound context without its own tuned binding uses the declared fallback; it never inherits the caller's.
      const reason = contextId ? 'this context binding has no tuned weights (rebinding with .in() needs its own binding)' : undefined;
      return this.base(null, [], [], 'untuned', reason);
    }
    const artifacts = new Map(this.manifest.artifacts.map(artifact => [artifact.node, artifact]));
    const requested: StackEntry[] = [];
    const families = this.manifest.memberships.filter(item => item.binding === entry.id && item.gate > 0)
      .sort((a, b) => a.family.localeCompare(b.family));
    for (const membership of families) {
      const family = this.manifest.families.find(item => item.family === membership.family && item.version === membership.version)!;
      requested.push({ id: family.artifact, scale: membership.gate, node: `family:${family.family}@${family.version}` });
    }
    const requires = new Map<string, readonly string[]>();
    for (const node of entry.ancestry) {
      const artifact = artifacts.get(node);
      if (!artifact) continue; // an absent level is a valid empty level
      const scale = this.manifest.scales[node] ?? 1;
      if (scale === 0) continue;
      requested.push({ id: artifact.artifact, scale, node });
      requires.set(artifact.artifact, artifact.requires);
    }
    if (this.status === 'degraded') return this.base(entry.id, requested, [], 'degraded', this.reason);
    if (this.staleBindings.has(entry.id))
      return this.base(entry.id, requested, [], 'stale', 'this definition changed since tuning; it runs untuned until retuned');
    const available = this.options.available;
    const skipped: { artifact: string; node: string; reason: string }[] = [];
    const dropped = new Set<string>();
    for (const item of requested) if (available && !available.has(item.id)) {
      dropped.add(item.id); skipped.push({ artifact: item.id, node: item.node, reason: 'not loaded by the serving backend' });
    }
    // Dependency closure: a child fit against a dropped parent version goes too.
    for (let changed = dropped.size > 0; changed;) {
      changed = false;
      for (const item of requested) if (!dropped.has(item.id) && (requires.get(item.id) ?? []).some(parent => dropped.has(parent))) {
        dropped.add(item.id); skipped.push({ artifact: item.id, node: item.node, reason: 'requires a dropped parent version' }); changed = true;
      }
    }
    const softStates = this.manifest.softStates.filter(item => item.binding === entry.id);
    if (softStates.length && !this.options.nativeState) {
      // A soft-state binding was trained with its states; without them it is an unvalidated mixture.
      const reason = 'this definition was tuned with soft instruction/skill states and the serving backend cannot import native state';
      if (this.options.strict) throw new WeightBindingError('weights-unavailable', reason);
      return this.base(entry.id, requested, softStates.map(item => ({ artifact: item.state, node: `soft:${item.kind}`, reason: 'backend has no native-state import' })), 'degraded', reason);
    }
    const soft = softStates.length ? { ...(softStates.find(item => item.kind === 'instructions')
      ? { instructions: (({ state, textSha256 }) => ({ state, textSha256 }))(softStates.find(item => item.kind === 'instructions')!) } : {}),
      skills: Object.fromEntries(softStates.filter(item => item.kind === 'skill').map(item => [item.target!, { state: item.state, textSha256: item.textSha256 }])) } : undefined;
    if (!dropped.size)
      return { release: this.release, program: this.programId, binding: entry.id, requested, effective: requested, skipped,
        status: this.status, ...(this.reason ? { reason: this.reason } : {}), ...(soft ? { soft } : {}),
        identity: stackIdentity(this.manifest.base, requested, soft), baseIdentity: stackIdentity(this.manifest.base, []) };
    if (this.options.strict)
      throw new WeightBindingError('weights-unavailable', `binding ${entry.id} cannot be served: ${skipped.map(item => `${item.node} (${item.reason})`).join(', ')}`);
    // A reduced stack never keeps soft states trained jointly with the dropped residuals.
    const reduced = requested.filter(item => !dropped.has(item.id));
    const reducedKey = JSON.stringify(reduced.map(item => item.id).sort());
    if (reduced.length && this.manifest.validatedReductions.some(item => JSON.stringify([...item].sort()) === reducedKey))
      return { release: this.release, program: this.programId, binding: entry.id, requested, effective: reduced, skipped,
        status: 'degraded', reason: 'running a validated reduced stack', identity: stackIdentity(this.manifest.base, reduced), baseIdentity: stackIdentity(this.manifest.base, []) };
    return this.base(entry.id, requested, skipped, 'degraded', 'adapters unavailable; running the base model');
  }
}

/** The binding of a definition's owning program among the runtime's releases. */
export function weightBindingFor(bindings: readonly WeightBinding[] | undefined, programId: string | undefined): WeightBinding | undefined {
  return programId ? bindings?.find(binding => binding.programId === programId) : undefined;
}
