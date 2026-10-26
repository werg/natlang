/**
 * Persistent site identities for tuning (plans/HIERARCHICAL_LORA_TUNING.md §4.4 "Site identity").
 *
 * A named definition's component key is its package-relative path, which is stable. An unlabeled inline `nl` site's
 * key is derived from its file revision and offsets, so any edit of its file renames every such site. The tuner keeps
 * its own registry (`.natlang/tune/sites.json`) from current component keys to persistent site IDs and carries a
 * site across edits only by an unambiguous structural match:
 *
 * 1. an explicit migration (`tune.migrations.json` beside the program: `{ "<site ID>": "<current component key>" }`);
 * 2. the same component key;
 * 3. the same file, instruction text and contract (moved within its file; equal counts pair in source order);
 * 4. the same file and contract with edited text, when exactly one old and one new site remain.
 *
 * Anything else gets a new site: a binding never silently inherits another site's adapters or data. Named
 * definitions move only by an explicit migration. Site IDs name bindings and lambda nodes; the runtime still resolves
 * by the current component key, so a stale release never serves an edited site.
 */
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import type { ProgramDescriptor } from '../adaptation/types.js';
import { tuneDirectory } from '../adaptation/weights/release.js';

export const SITES_SCHEMA = 'natlang.tune-sites/v1' as const;
export type SiteEntry = { readonly site: string; readonly component: string; readonly origin: 'named' | 'authored-inline';
  readonly file: string; readonly baselineHash: string; readonly contractHash: string };
export type SiteRegistry = { readonly schema: typeof SITES_SCHEMA; readonly entries: readonly SiteEntry[] };
export type SiteMigration = { readonly site: string; readonly from: string; readonly to: string;
  readonly how: 'explicit' | 'moved' | 'edited' };
export type SiteAssignment = { readonly registry: SiteRegistry;
  /** Current component key → persistent site ID (only where they differ is it informative). */
  readonly sites: Readonly<Record<string, string>>;
  readonly migrations: readonly SiteMigration[];
  /** Previous sites that matched no current component unambiguously (removed, or ambiguous edits). */
  readonly unmatched: readonly string[] };

const entryOf = (component: ProgramDescriptor['components'][number], site: string): SiteEntry => ({ site, component: component.key,
  origin: component.origin as SiteEntry['origin'], file: component.source!.path, baselineHash: component.baselineHash, contractHash: component.contractHash });

/** Assign persistent site IDs to the program's bound components. Pure. */
export function assignSites(program: ProgramDescriptor, previous?: SiteRegistry, explicit: Readonly<Record<string, string>> = {}): SiteAssignment {
  const current = program.components.filter(component => component.origin !== 'program' && component.source)
    .sort((a, b) => a.source!.path.localeCompare(b.source!.path) || a.source!.start - b.source!.start);
  const old = new Map((previous?.entries ?? []).map(entry => [entry.site, entry]));
  const assigned = new Map<string, string>(), taken = new Set<string>(), migrations: SiteMigration[] = [];
  const claim = (component: string, site: string, how?: SiteMigration['how']) => {
    assigned.set(component, site); taken.add(site);
    const from = old.get(site)?.component;
    if (how && from !== undefined && from !== component) migrations.push({ site, from, to: component, how });
  };
  const byKey = new Map(current.map(component => [component.key, component]));
  for (const [site, key] of Object.entries(explicit)) {
    if (!old.has(site)) throw new Error(`tune.migrations.json names site ${site}, which is not in the site registry`);
    if (!byKey.has(key)) throw new Error(`tune.migrations.json maps ${site} to ${key}, which is not a component of this program; update or remove that entry`);
    if (assigned.has(key) || taken.has(site)) throw new Error(`tune.migrations.json maps ${key} or ${site} twice`);
    claim(key, site, 'explicit');
  }
  const free = () => [...old.values()].filter(entry => !taken.has(entry.site));
  const open = () => current.filter(component => !assigned.has(component.key));
  for (const entry of free()) {
    const component = byKey.get(entry.component);
    // The same key at the same place: a site, unless a different contract and text now sit at that key.
    if (component && !assigned.has(component.key) && (component.contractHash === entry.contractHash || component.baselineHash === entry.baselineHash))
      claim(component.key, entry.site);
  }
  const pair = (signature: (item: { file: string; baselineHash: string; contractHash: string }) => string, how: SiteMigration['how'], exactCounts: boolean) => {
    const before = new Map<string, SiteEntry[]>(), after = new Map<string, typeof current>();
    for (const entry of free()) if (entry.origin === 'authored-inline') before.set(signature(entry), [...before.get(signature(entry)) ?? [], entry]);
    for (const component of open()) if (component.origin === 'authored-inline') {
      const key = signature({ file: component.source!.path, baselineHash: component.baselineHash, contractHash: component.contractHash });
      after.set(key, [...after.get(key) ?? [], component]);
    }
    for (const [key, entries] of before) {
      const components = after.get(key) ?? [];
      if (!components.length || (exactCounts ? entries.length !== components.length : entries.length !== 1 || components.length !== 1)) continue;
      // Old entries are kept in registry (source) order; equal counts pair positionally.
      entries.forEach((entry, index) => claim(components[index]!.key, entry.site, how));
    }
  };
  pair(item => JSON.stringify([item.file, item.baselineHash, item.contractHash]), 'moved', true);
  pair(item => JSON.stringify([item.file, item.contractHash]), 'edited', false);
  for (const component of open()) {
    let site = component.key;
    // Never reuse an old site ID that did not match: that would inherit another site's adapters.
    for (let suffix = 2; taken.has(site) || old.has(site); suffix++) site = `${component.key}#${suffix}`;
    claim(component.key, site);
  }
  const unmatched = free().map(entry => entry.site).sort();
  const entries = current.map(component => entryOf(component, assigned.get(component.key)!));
  return { registry: { schema: SITES_SCHEMA, entries }, migrations,
    sites: Object.fromEntries(entries.map(entry => [entry.component, entry.site])), unmatched };
}

/** Read the registry and the author's explicit migrations, assign, and persist the new registry. */
export function programSites(root: string, program: ProgramDescriptor): SiteAssignment {
  const path = join(tuneDirectory(root), 'sites.json');
  const previous = existsSync(path) ? JSON.parse(readFileSync(path, 'utf8')) as SiteRegistry : undefined;
  if (previous && previous.schema !== SITES_SCHEMA) throw new Error(`${path} is not a ${SITES_SCHEMA} registry`);
  const migrationsPath = join(root, 'tune.migrations.json');
  const explicit = existsSync(migrationsPath) ? JSON.parse(readFileSync(migrationsPath, 'utf8')) as Record<string, string> : {};
  const assignment = assignSites(program, previous, explicit);
  mkdirSync(tuneDirectory(root), { recursive: true });
  writeFileSync(path + '.tmp', JSON.stringify(assignment.registry, null, 1) + '\n');
  renameSync(path + '.tmp', path);
  return assignment;
}
