/**
 * Build keys, reverse dependencies and invalidation for `natlang tune` (plans/HIERARCHICAL_LORA_TUNING.md §8).
 *
 * Keys are acyclic: source/data/configuration first, then trained artifacts, then exports, then the release. Semantic
 * digests never include generated adapters or manifests. Raw collected evidence survives edits; only the rows and
 * artifacts whose semantics changed are invalidated, and shared ancestors trained on changed semantics are dirty
 * together with every residual that was fit against them.
 */
import { createHash } from 'node:crypto';
import type { ProgramDescriptor } from '../adaptation/types.js';
import type { WeightManifest } from '../adaptation/weights/manifest.js';

const sha = (value: unknown) => createHash('sha256').update(JSON.stringify(value)).digest('hex');

/** Everything a root's trajectories depend on besides the lambdas it reached: non-instruction sources (helpers,
 * services, types, TS modules) and program-level policy. `.nl` files are covered by binding semantic hashes. */
export function sharedSourceHash(program: ProgramDescriptor): string {
  return sha(Object.entries(program.sources).filter(([path]) => !path.endsWith('.nl')).sort(([a], [b]) => a.localeCompare(b)));
}

export type Invalidation = {
  /** Bindings whose semantics changed, that disappeared, appeared, or moved in the scope tree. */
  readonly changed: readonly string[]; readonly removed: readonly string[]; readonly added: readonly string[]; readonly moved: readonly string[];
  /** Bindings whose site continues under a new component key with unchanged semantics: no refit, but the release must
   * be resealed for the current keys (training receipts are reused). */
  readonly rekeyed: readonly string[];
  /** Nodes to refit: the changed bindings' own nodes and every shared ancestor trained on changed semantics. */
  readonly dirty: readonly string[];
  /** Nodes fit against a dirty parent version: refit after their parents (reverse dependencies). */
  readonly dependents: readonly string[];
  /** Whether nothing changed (the release is fresh). */
  readonly fresh: boolean;
};

/** Compare a previous release's bindings with the current program scaffold. */
export function invalidation(previous: WeightManifest, current: WeightManifest): Invalidation {
  const before = new Map(previous.bindings.map(entry => [entry.id, entry]));
  const after = new Map(current.bindings.map(entry => [entry.id, entry]));
  const changed: string[] = [], removed: string[] = [], added: string[] = [], moved: string[] = [], rekeyed: string[] = [];
  for (const [id, entry] of before) {
    const now = after.get(id);
    if (!now) { removed.push(id); continue; }
    if (now.semanticHash !== entry.semanticHash) changed.push(id);
    else if (now.component !== entry.component) rekeyed.push(id);
    if (JSON.stringify(now.ancestry) !== JSON.stringify(entry.ancestry)) moved.push(id);
  }
  for (const id of after.keys()) if (!before.has(id)) added.push(id);
  const dirty = new Set<string>();
  // A changed or moved binding's whole path (before and after) saw different training semantics or population.
  for (const id of [...changed, ...moved]) for (const node of [...(before.get(id)?.ancestry ?? []), ...(after.get(id)?.ancestry ?? [])]) dirty.add(node);
  for (const id of removed) for (const node of before.get(id)!.ancestry) dirty.add(node);
  for (const id of added) for (const node of after.get(id)!.ancestry) dirty.add(node);
  // Reverse dependencies: every node below a dirty shared node was fit against that parent version.
  const children = new Map<string, string[]>();
  for (const node of current.tree) if (node.parent) children.set(node.parent, [...(children.get(node.parent) ?? []), node.id]);
  const dependents = new Set<string>();
  const visit = (node: string) => { for (const child of children.get(node) ?? []) if (!dirty.has(child) && !dependents.has(child)) { dependents.add(child); visit(child); } };
  for (const node of dirty) visit(node);
  const currentNodes = new Set(current.tree.map(node => node.id));
  const keep = (nodes: Iterable<string>) => [...nodes].filter(node => currentNodes.has(node)).sort();
  return { changed: changed.sort(), removed: removed.sort(), added: added.sort(), moved: moved.sort(), rekeyed: rekeyed.sort(),
    dirty: keep(dirty), dependents: keep(dependents), fresh: !changed.length && !removed.length && !added.length && !moved.length && !rekeyed.length };
}

/** A root's collected trajectories stay valid while every binding they reached keeps its semantics and the shared
 * (non-instruction) sources are unchanged. */
export function rootDataValid(record: { readonly bindings: Readonly<Record<string, string>>; readonly sharedSources: string },
  current: WeightManifest, sharedSources: string): boolean {
  if (record.sharedSources !== sharedSources) return false;
  const semantic = new Map(current.bindings.map(entry => [entry.id, entry.semanticHash]));
  return Object.entries(record.bindings).every(([id, hash]) => semantic.get(id) === hash);
}

/** What training depends on in a scaffold: the tree, bindings by persistent ID with their ancestry and semantics, and
 * the declared scales/families. Component keys, definition IDs and the program build hash only bind a release to the
 * current source, so an edit that only renames a site reuses its training receipt. */
export function trainingScaffoldKey(manifest: WeightManifest): string {
  const { digest: _digest, program: _program, bindings, ...rest } = manifest;
  return sha([rest, bindings.map(({ component: _component, definitionId: _definition, site: _site, ...binding }) => binding)]);
}
