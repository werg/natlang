/**
 * The weight-binding manifest (`natlang.adapters.json`, plans/HIERARCHICAL_LORA_TUNING.md §3, §6): which residual
 * weight adapters each bound definition of a program runs with. A binding's stack is resolved from the callee, never
 * from its caller: families (gated), then its lexical scope ancestry from the program root to its own lambda node.
 *
 * Generated data, not model-visible prompt content. Validation is exact; a malformed manifest is an error, not a
 * silently untuned program.
 */
import { fingerprint, immutable } from '../identity.js';
import type { NativeModelIdentity } from '../../native/neuralese-store.js';

export const WEIGHTS_SCHEMA = 'natlang.adapters/v1' as const;
const DIGEST_DOMAIN = 'natlang.adapters/v1';

export type ScopeLevel = 'tree' | 'folder' | 'lambda';
/** One node of the lexical adapter-scope tree; `parent` is null only for the program root. */
export type ScopeNode = { readonly id: string; readonly level: ScopeLevel; readonly path: string; readonly parent: string | null };
/** A bound definition: its component, its context binding (null: its file context) and ordered scope ancestry. */
export type WeightBindingEntry = { readonly id: string; readonly component: string; readonly definitionId: string | null;
  readonly contextId: string | null; readonly origin: 'named' | 'authored-inline';
  /** Root first, the binding's own lambda node last. */
  readonly ancestry: readonly string[];
  /** The component's baseline and contract hashes the adapters were trained against. */
  readonly semanticHash: string };
/** A trained residual for one scope node: an archived nnt1 coefficient artifact. `requires` names the exact parent
 * artifacts it was fit against, so a fallback never keeps a child without the parent version it assumed. */
export type WeightArtifact = { readonly node: string; readonly artifact: string; readonly rank: number;
  readonly parameterization: 'native-subspace' | 'free-lora'; readonly requires: readonly string[] };
/** An immutable published family version for this base; its amplitude is fixed (§5.4). */
export type FamilyVersion = { readonly family: string; readonly version: string; readonly artifact: string; readonly amplitudeHex: string };
/** A static per-binding family membership with its learned, bounded gate. */
export type FamilyMembership = { readonly binding: string; readonly family: string; readonly version: string; readonly gate: number };

export type WeightManifest = {
  readonly schema: typeof WEIGHTS_SCHEMA;
  readonly program: { readonly id: string; readonly buildHash: string };
  /** The exact base the adapters were trained on. */
  readonly base: NativeModelIdentity;
  readonly tree: readonly ScopeNode[];
  readonly bindings: readonly WeightBindingEntry[];
  readonly artifacts: readonly WeightArtifact[];
  readonly families: readonly FamilyVersion[];
  readonly memberships: readonly FamilyMembership[];
  /** Fixed deployment/ablation scale per node (normally 1 or 0); absent means 1. */
  readonly scales: Readonly<Record<string, number>>;
  /** Reduced stacks (sorted artifact IDs) validated as usable when some artifacts are unavailable. */
  readonly validatedReductions: readonly (readonly string[])[];
  /** What runs when a binding's stack cannot be served. Only the compatible base executor is declared today. */
  readonly fallback: 'base';
  /** Build receipts (data, training, export); opaque here. */
  readonly receipts: Readonly<Record<string, unknown>>;
  readonly digest: string;
};

export class WeightManifestError extends Error {
  constructor(message: string) { super(`weights-manifest: ${message}`); this.name = 'WeightManifestError'; }
}

export function manifestDigest(manifest: Omit<WeightManifest, 'digest'> | WeightManifest): string {
  const { digest: _digest, ...rest } = manifest as WeightManifest;
  return fingerprint(rest, DIGEST_DOMAIN);
}

/** Seal a manifest: compute its digest and freeze it. */
export function sealManifest(manifest: Omit<WeightManifest, 'digest'>): WeightManifest {
  const sealed = { ...manifest, digest: manifestDigest(manifest) } as WeightManifest;
  validateManifest(sealed);
  return immutable(sealed);
}

const isString = (value: unknown): value is string => typeof value === 'string' && value.length > 0;
const IDENTITY_KEYS = ['architecture', 'checkpoint_sha256', 'embedding_weight_sha256', 'tokenizer_fingerprint'] as const;

/** Exact structural validation. Throws on the first problem with the field that is wrong. */
export function validateManifest(value: unknown): asserts value is WeightManifest {
  const fail = (message: string): never => { throw new WeightManifestError(message); };
  if (!value || typeof value !== 'object') fail('not an object');
  const manifest = value as WeightManifest;
  if (manifest.schema !== WEIGHTS_SCHEMA) fail(`schema must be ${WEIGHTS_SCHEMA}`);
  if (!isString(manifest.program?.id) || !isString(manifest.program?.buildHash)) fail('program needs id and buildHash');
  if (!manifest.base || IDENTITY_KEYS.some(key => !isString(manifest.base[key])) || Object.keys(manifest.base).length !== IDENTITY_KEYS.length)
    fail('base must be an exact native model identity');
  const nodes = new Map<string, ScopeNode>();
  for (const node of manifest.tree ?? fail('tree missing')) {
    if (!isString(node.id) || !['tree', 'folder', 'lambda'].includes(node.level) || typeof node.path !== 'string') fail(`bad scope node ${node?.id}`);
    if (nodes.has(node.id)) fail(`duplicate scope node ${node.id}`);
    nodes.set(node.id, node);
  }
  const roots = [...nodes.values()].filter(node => node.parent === null);
  if (nodes.size && (roots.length !== 1 || roots[0]!.level !== 'tree')) fail('the scope tree needs exactly one program root of level tree');
  for (const node of nodes.values()) if (node.parent !== null && !nodes.has(node.parent)) fail(`scope node ${node.id} has unknown parent`);
  const bindings = new Set<string>(), definitions = new Set<string>();
  for (const binding of manifest.bindings ?? fail('bindings missing')) {
    if (!isString(binding.id) || !isString(binding.component) || !isString(binding.semanticHash)) fail('binding needs id, component, semanticHash');
    if (bindings.has(binding.id)) fail(`duplicate binding ${binding.id}`);
    bindings.add(binding.id);
    const key = JSON.stringify([binding.definitionId ?? binding.component, binding.contextId]);
    if (definitions.has(key)) fail(`two bindings for definition ${binding.definitionId ?? binding.component} in one context`);
    definitions.add(key);
    const ancestry = binding.ancestry ?? [];
    if (!ancestry.length) fail(`binding ${binding.id} has no scope ancestry`);
    ancestry.forEach((id, index) => {
      const node = nodes.get(id) ?? fail(`binding ${binding.id} names unknown scope node ${id}`);
      if (node.parent !== (index ? ancestry[index - 1] : null)) fail(`binding ${binding.id} ancestry is not a root-to-leaf path at ${id}`);
    });
    if (nodes.get(ancestry.at(-1)!)!.level !== 'lambda') fail(`binding ${binding.id} must end at its lambda node`);
  }
  const byNode = new Set<string>(), artifacts = new Set<string>();
  for (const artifact of manifest.artifacts ?? fail('artifacts missing')) {
    if (!nodes.has(artifact.node)) fail(`artifact for unknown node ${artifact.node}`);
    if (byNode.has(artifact.node)) fail(`node ${artifact.node} has two artifacts`);
    if (!isString(artifact.artifact) || !Number.isInteger(artifact.rank) || artifact.rank < 1) fail(`artifact on ${artifact.node} needs id and positive rank`);
    if (!['native-subspace', 'free-lora'].includes(artifact.parameterization)) fail(`artifact on ${artifact.node} has unknown parameterization`);
    byNode.add(artifact.node); artifacts.add(artifact.artifact);
  }
  for (const artifact of manifest.artifacts) for (const parent of artifact.requires ?? fail('requires missing'))
    if (!artifacts.has(parent)) fail(`artifact on ${artifact.node} requires ${parent}, which this release does not carry`);
  const families = new Set<string>();
  for (const family of manifest.families ?? fail('families missing')) {
    if (!isString(family.family) || !isString(family.version) || !isString(family.artifact)) fail('family version needs family, version, artifact');
    // Python float.hex(): the exact fixed amplitude, positive (a near-zero family is absent, not published).
    if (typeof family.amplitudeHex !== 'string' || !/^0x[01]\.[0-9a-f]+p[+-]\d+$/i.test(family.amplitudeHex) || /^0x0\.0+p/i.test(family.amplitudeHex))
      fail(`family ${family.family} amplitude must be a positive float.hex() value`);
    const key = family.family + '@' + family.version;
    if (families.has(key)) fail(`duplicate family version ${key}`);
    families.add(key);
  }
  for (const membership of manifest.memberships ?? fail('memberships missing')) {
    if (!bindings.has(membership.binding)) fail(`membership for unknown binding ${membership.binding}`);
    if (!families.has(membership.family + '@' + membership.version)) fail(`membership names unpublished family ${membership.family}@${membership.version}`);
    if (!Number.isFinite(membership.gate) || membership.gate < 0) fail(`family gate for ${membership.binding} must be finite and nonnegative`);
  }
  for (const [node, scale] of Object.entries(manifest.scales ?? fail('scales missing')))
    if (!nodes.has(node) || !Number.isFinite(scale)) fail(`scale for ${node} must name a node and be finite`);
  if (manifest.fallback !== 'base') fail('fallback must be base');
  if (!Array.isArray(manifest.validatedReductions)) fail('validatedReductions missing');
  if (manifest.digest !== manifestDigest(manifest)) fail('digest does not match content');
}

export function parseManifest(text: string): WeightManifest {
  let value: unknown;
  try { value = JSON.parse(text); } catch (error) { throw new WeightManifestError(`not JSON: ${(error as Error).message}`); }
  validateManifest(value);
  return immutable(value);
}
