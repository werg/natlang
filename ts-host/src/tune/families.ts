/**
 * Cross-program family learning (plans/HIERARCHICAL_LORA_TUNING.md §5.4, §5.5): `natlang tune --families <programs...>`.
 *
 * 1. Every program must have an active release on one exact base; families never mix bases.
 * 2. Discovery (crisp, `natlang_neuralese.tune.families discover`) compares the bindings' effective deltas across
 *    programs with gauge-invariant Gram products and returns threshold clusters and author tags
 *    (`tune.families.json` beside a program: `{ "<family>": ["<source path of a definition>", ...] }`).
 * 3. `proposeFamilies` (pluggable: crisp clusters + tags by default, or applications/tuner/proposeFamilies.nl) decides
 *    which families to fit. Only discovered members can be named.
 * 4. Each family is fit on its members' TRAINING rows only, against the frozen base with private residuals absent,
 *    normalized to its fixed amplitude and published as an immutable version in the per-base family store
 *    (`<natlang config>/tune/families/<base digest>/`). A name already published keeps its amplitude.
 * 5. Each member program is refit through the normal pipeline with the family frozen and a gate per member binding
 *    (initial 1), measured against its previous release on development roots (paired, by root), and activated only
 *    within the placement envelope; the previous release stays for rollback. Negative transfer is reported.
 *
 * Held programs are simply not passed: nothing outside the given programs' training rows enters discovery or fits.
 */
import { createHash } from 'node:crypto';
import { copyFileSync, existsSync, mkdirSync, readdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import type { ProgramDescriptor } from '../adaptation/types.js';
import { canonical } from '../adaptation/identity.js';
import { sealManifest, type FamilyMembership, type FamilyVersion, type WeightManifest } from '../adaptation/weights/manifest.js';
import { activateRelease, readActive, storeRelease, tuneDirectory } from '../adaptation/weights/release.js';
import { defaultNatlangConfigDirectory } from '../package/store.js';
import { pluggable, type PluggableSetting } from '../runtime/pluggable.js';
import { rootSuccess, type ArmResult } from './evaluate.js';
import { evaluateRelease, ggufExports, runStudentModule, saveArm, trainRelease, type TuneData, type TuneOptions } from './pipeline.js';
import { pairedComparison } from './stats.js';
import { copyNativeReleaseClosure } from '../native/release-artifact-closure.js';

export type FamilyProgram = { readonly options: TuneOptions; readonly data: TuneData; readonly release: WeightManifest; readonly releaseDir: string };
export type FamilyMember = readonly [program: string, binding: string];
export type FamilyProposal = { readonly name: string; readonly members: readonly FamilyMember[]; readonly source: 'cluster' | 'tag' | 'nl' };
export type Discovery = { readonly schema: 'natlang.tune-family-discovery/v1'; readonly members: readonly FamilyMember[];
  readonly similarities: readonly { a: FamilyMember; b: FamilyMember; cosine: number }[];
  readonly clusters: readonly (readonly FamilyMember[])[]; readonly tags: Readonly<Record<string, readonly FamilyMember[]>> };
export type PublishedFamily = FamilyVersion & { readonly members: readonly string[]; readonly base: string };
export type FamiliesSettings = { readonly threshold?: number; readonly minPrograms?: number; readonly mode?: PluggableSetting; readonly every?: number;
  readonly recipe?: Readonly<Record<string, unknown>> };
export type ProgramOutcome = { readonly program: string; readonly families: readonly string[]; readonly release?: string;
  readonly previous: string; readonly development?: { previous: number; refit: number; roots: number; delta: number; interval: readonly [number, number]; underpowered: boolean };
  readonly activated: boolean; readonly negativeTransfer: boolean; readonly reason?: string };
/** Zero-shot transfer of one published family to a program that did not train it (§5.5 step 4): the family at gate 1 on
 * every binding of the program's current release, without refitting, paired on development roots. Diagnostic only. */
export type TransferOutcome = { readonly family: string; readonly version: string; readonly program: string;
  readonly kind: 'non-member' | 'held'; readonly release?: string;
  readonly development?: { current: number; withFamily: number; roots: number; delta: number; interval: readonly [number, number]; underpowered: boolean };
  readonly reason?: string };
export type FamiliesReport = { readonly schema: 'natlang.tune-families/v1'; readonly base: string; readonly discovery: Discovery;
  readonly transfer: readonly TransferOutcome[];
  readonly proposals: readonly FamilyProposal[]; readonly published: readonly PublishedFamily[]; readonly absent: readonly string[];
  readonly programs: readonly ProgramOutcome[] };

const sha = (value: unknown) => createHash('sha256').update(typeof value === 'string' ? value : canonical(value)).digest('hex');
const writeJson = (path: string, value: unknown) => {
  mkdirSync(dirname(path), { recursive: true }); writeFileSync(path + '.tmp', JSON.stringify(value, null, 1)); renameSync(path + '.tmp', path);
};
/** A program's namespace in family rows and bindings: no `/`, so `<namespace>/<binding>` splits unambiguously. */
export const programNamespace = (program: ProgramDescriptor) => 'p' + sha(program.id).slice(0, 12);
export const familyStore = (base: unknown, environment: NodeJS.ProcessEnv = process.env) =>
  join(defaultNatlangConfigDirectory(environment), 'tune', 'families', sha(base).slice(0, 24));

/** Author tags beside a program: family name → definition source paths, resolved to that release's binding IDs. */
export function authorTags(root: string, release: WeightManifest): Record<string, string[]> {
  const path = join(root, 'tune.families.json');
  if (!existsSync(path)) return {};
  const tags = JSON.parse(readFileSync(path, 'utf8')) as Record<string, string[]>;
  const pathOf = (component: string) => decodeURIComponent(component.split('::')[1] ?? '');
  const result: Record<string, string[]> = {};
  for (const [family, paths] of Object.entries(tags)) {
    if (!/^[\w.-]+$/.test(family)) throw new Error(`tune.families.json: family name ${JSON.stringify(family)} must be letters, digits, ., _ or -`);
    for (const source of paths) {
      const bindings = release.bindings.filter(binding => pathOf(binding.component) === source.replace(/\\/g, '/'));
      if (!bindings.length) throw new Error(`tune.families.json: ${source} is not a tuned definition of this program; tag a .nl file the release binds`);
      (result[family] ??= []).push(...bindings.map(binding => binding.id));
    }
  }
  return result;
}

/** The crisp proposal: every author tag, then every discovered cluster not already covered by a tag. */
export function crispProposals(discovery: Discovery): FamilyProposal[] {
  const proposals: FamilyProposal[] = Object.entries(discovery.tags).filter(([, members]) => members.length >= 2)
    .map(([name, members]) => ({ name, members, source: 'tag' }));
  const covered = new Set(proposals.flatMap(item => item.members.map(member => member.join('/'))));
  for (const cluster of discovery.clusters) {
    const members = cluster.filter(member => !covered.has(member.join('/')));
    if (new Set(members.map(([program]) => program)).size >= 2)
      proposals.push({ name: 'family-' + sha(members).slice(0, 10), members, source: 'cluster' });
  }
  return proposals;
}

export type ProposeFunction = (discovery: string) => unknown;

/** Which families to fit. An answer may only name discovered members; anything malformed falls back to crisp. */
export async function proposeFamilies(discovery: Discovery, mode: PluggableSetting, nl?: ProposeFunction): Promise<FamilyProposal[]> {
  const known = new Set(discovery.members.map(member => member.join('/')));
  return pluggable<[], FamilyProposal[]>({
    crisp: () => crispProposals(discovery),
    nl: async () => {
      if (!nl) throw new Error('proposeFamilies: the nl policy needs applications/tuner/proposeFamilies.nl');
      const answer = await nl(JSON.stringify({ members: discovery.members.map(member => member.join('/')),
        similarities: discovery.similarities.map(item => ({ a: item.a.join('/'), b: item.b.join('/'), cosine: Math.round(item.cosine * 1000) / 1000 })),
        clusters: discovery.clusters.map(cluster => cluster.map(member => member.join('/'))),
        tags: Object.fromEntries(Object.entries(discovery.tags).map(([name, members]) => [name, members.map(member => member.join('/'))])) }));
      const families = (answer as { families?: unknown } | null)?.families;
      const valid = Array.isArray(families) && families.every(item => typeof item?.name === 'string' && /^[\w.-]+$/.test(item.name)
        && Array.isArray(item.members) && item.members.length >= 2 && item.members.every((member: unknown) => typeof member === 'string' && known.has(member)));
      if (!valid) return crispProposals(discovery);
      return (families as { name: string; members: string[] }[]).map(item => ({ name: item.name, source: 'nl' as const,
        members: item.members.map(member => member.split('/') as unknown as FamilyMember) }));
    } }, mode, { name: 'proposeFamilies', default: 'crisp', serve: 'crisp',
    same: (a, b) => canonical(a.map(item => item.members)) === canonical(b.map(item => item.members)) })();
}

function readStore(store: string): PublishedFamily[] {
  const path = join(store, 'versions.json');
  return existsSync(path) ? JSON.parse(readFileSync(path, 'utf8')) as PublishedFamily[] : [];
}

/** Run the families pass over programs that share one base. */
export async function runFamilies(programs: readonly FamilyProgram[], settings: FamiliesSettings & { store?: string;
  nl?: ProposeFunction; log?: (line: string) => void;
  /** Programs kept outside discovery and fitting (no rows, no similarity): only measured for transfer. */
  held?: readonly FamilyProgram[] } = {}): Promise<FamiliesReport> {
  const log = settings.log ?? (() => {});
  if (programs.length < 1) throw new Error('natlang tune --families needs at least one tuned program');
  const base = programs[0]!.release.base;
  for (const program of [...programs, ...settings.held ?? []]) if (canonical(program.release.base) !== canonical(base))
    throw new Error(`${program.options.root} is tuned on another base; families never mix bases (retune it for ${base.architecture})`);
  const store = settings.store ?? familyStore(base);
  const byNamespace = new Map(programs.map(program => [programNamespace(program.options.program), program]));
  if (byNamespace.size !== programs.length) throw new Error('the same program was given twice');
  const student = programs[0]!.options.student;
  // 2. Discovery over the active releases.
  const work = join(store, 'work', sha(programs.map(program => [program.options.program.id, program.release.digest])).slice(0, 16));
  mkdirSync(work, { recursive: true });
  const tags: Record<string, FamilyMember[]> = {};
  for (const [namespace, program] of byNamespace)
    for (const [family, bindings] of Object.entries(authorTags(program.options.root, program.release)))
      (tags[family] ??= []).push(...bindings.map(binding => [namespace, binding] as const));
  const discoveryPlan = join(work, 'discover.json');
  writeJson(discoveryPlan, { programs: [...byNamespace].map(([id, program]) => ({ id, release: program.releaseDir })),
    threshold: settings.threshold ?? .5, minPrograms: settings.minPrograms ?? 2, tags });
  await runStudentModule(student, 'natlang_neuralese.tune.families', ['discover', '--plan', discoveryPlan, '--out', work]);
  const discovery = JSON.parse(readFileSync(join(work, 'discovery.json'), 'utf8')) as Discovery;
  // 3. Proposals.
  const proposals = await proposeFamilies(discovery, settings.mode, settings.nl);
  log(`natlang tune: ${discovery.members.length} tuned bindings, ${proposals.length} family proposal(s)`);
  // 4. Fit and publish each family on training rows only.
  const published: PublishedFamily[] = [], absent: string[] = [];
  const existing = readStore(store);
  for (const proposal of proposals) {
    const members = [...new Set(proposal.members.map(([namespace, binding]) => `${namespace}/${binding}`))].sort();
    const rows = proposal.members.flatMap(([namespace, binding]) => byNamespace.get(namespace)!.data.trainRows
      .filter(turn => turn.binding === binding && turn.partition === 'train')
      .map(turn => ({ ...turn, binding: `${namespace}/${binding}`, root: `${namespace}/${turn.root}` })));
    if (!rows.length) { absent.push(proposal.name); continue; }
    const previous = existing.filter(item => item.family === proposal.name).at(-1);
    const recipe = { rank: 8, steps: 8, batch_rows: 8, ...settings.recipe };
    // The fit is keyed by what it learns from; the amplitude it is normalized to is fixed by the family's first fit.
    const key = sha({ family: proposal.name, members, rows: rows.map(row => [row.root, row.call, row.turn, sha(row.reply)]), recipe,
      student: student.base }).slice(0, 24);
    const job = join(store, 'jobs', key);
    if (!existsSync(join(job, 'result.json'))) {
      mkdirSync(job, { recursive: true });
      writeFileSync(join(job, 'rows.jsonl'), rows.map(row => JSON.stringify(row)).join('\n') + '\n');
      writeJson(join(job, 'plan.json'), { family: proposal.name, members, rows: join(job, 'rows.jsonl'), recipe,
        student: { name: student.name, checkpoint: student.checkpoint, engine: student.engine ?? {}, moduleMap: student.moduleMap,
          ...(student.hfConfigDir ? { hfConfigDir: student.hfConfigDir } : {}), ...(student.baseGgufSha256 ? { baseGgufSha256: student.baseGgufSha256 } : {}) },
        exports: ggufExports(programs[0]!.options.settings, programs[0]!.options.student),
        // A name already published keeps its fixed amplitude (§5.4): the new version is normalized to it.
        ...(previous ? { rho: previous.amplitudeHex } : {}) });
      log(`natlang tune: fitting family ${proposal.name} on ${rows.length} training turns of ${members.length} bindings`);
      await runStudentModule(student, 'natlang_neuralese.tune.families', ['fit', '--plan', join(job, 'plan.json'), '--out', job]);
    }
    const result = JSON.parse(readFileSync(join(job, 'result.json'), 'utf8')) as { absent?: true; family: string; version: string;
      artifact: string; amplitudeHex: string; exports?: FamilyVersion['exports'] };
    if (result.absent) { absent.push(proposal.name); continue; }
    // Publish immutably: the artifact closure and exports into the store, the version record appended once.
    for (const folder of ['artifacts', 'exports']) if (existsSync(join(job, folder))) {
      mkdirSync(join(store, folder), { recursive: true });
      for (const file of readdirSync(join(job, folder))) if (!file.endsWith('.peft') && !existsSync(join(store, folder, file)))
        copyFileSync(join(job, folder, file), join(store, folder, file));
    }
    const version: PublishedFamily = { family: result.family, version: result.version, artifact: result.artifact,
      amplitudeHex: result.amplitudeHex, members, base: sha(base).slice(0, 24), ...(result.exports ? { exports: result.exports } : {}) };
    const current = readStore(store);
    if (!current.some(item => item.family === version.family && item.version === version.version)) writeJson(join(store, 'versions.json'), [...current, version]);
    published.push(version);
  }
  // 5. Refit each member program with its families frozen; activate within the envelope.
  const outcomes: ProgramOutcome[] = [];
  for (const [namespace, program] of byNamespace) {
    const mine = published.filter(family => family.members.some(member => member.startsWith(namespace + '/')));
    if (!mine.length) continue;
    outcomes.push(await refitProgram(program, namespace, mine, store, log));
  }
  // 6. Transfer: every published family on the programs that did not train it, and on held programs.
  const transfer: TransferOutcome[] = [];
  const targets = [...[...byNamespace].map(([namespace, program]) => ({ namespace, program, kind: 'non-member' as const })),
    ...(settings.held ?? []).map(program => ({ namespace: programNamespace(program.options.program), program, kind: 'held' as const }))];
  for (const family of published) for (const { namespace, program, kind } of targets) {
    if (family.members.some(member => member.startsWith(namespace + '/'))) continue;
    transfer.push(await measureTransfer(program, family, kind, store, log));
  }
  const report: FamiliesReport = { schema: 'natlang.tune-families/v1', base: sha(base).slice(0, 24), discovery, proposals, published, absent,
    programs: outcomes, transfer };
  writeJson(join(work, 'report.json'), report);
  for (const program of [...programs, ...settings.held ?? []]) writeJson(join(tuneDirectory(program.options.root), 'families', 'latest.json'), report);
  markFamiliesRun(store, programs.map(program => program.options.root));
  return report;
}

async function refitProgram(program: FamilyProgram, namespace: string, families: readonly PublishedFamily[], store: string,
  log: (line: string) => void): Promise<ProgramOutcome> {
  const { options, data, release } = program;
  const bindings = new Set(data.scaffold.bindings.map(binding => binding.id));
  const memberships: FamilyMembership[] = families.flatMap(family => family.members
    .filter(member => member.startsWith(namespace + '/')).map(member => member.slice(namespace.length + 1))
    .filter(binding => bindings.has(binding)).map(binding => ({ binding, family: family.family, version: family.version, gate: 1 })));
  const base = { program: options.program.id, families: families.map(family => `${family.family}@${family.version}`), previous: release.digest };
  if (!memberships.length) return { ...base, activated: false, negativeTransfer: false, reason: 'the program changed since its release; no member binding remains' };
  const { digest: _digest, ...scaffold } = data.scaffold;
  const manifest = sealManifest({ ...scaffold, families: families.map(({ members: _m, base: _b, ...version }) => version), memberships });
  const { manifest: refit } = await trainRelease(options, data, { manifest, label: 'families', artifactDirs: [join(store, 'artifacts')] }, [], log);
  const development = data.examples.filter(example => example.partition === 'development');
  if (!development.length) return { ...base, release: refit.digest, activated: false, negativeTransfer: false, reason: 'no development roots to measure the refit' };
  const previousArm = await evaluateRelease(options, data, release, development, 'hierarchy', 'development');
  const refitArm = await evaluateRelease(options, data, refit, development, 'families', 'development');
  saveArm(options.root, previousArm); saveArm(options.root, refitArm);
  const rate = (arm: ArmResult) => arm.roots.reduce((sum, root) => sum + rootSuccess(root), 0) / Math.max(arm.roots.length, 1);
  const comparison = pairedComparison(previousArm, refitArm);
  // Paired by root: the refit's success minus the previous release's, with a component-clustered interval.
  const delta = comparison.effect;
  const exercised = refitArm.roots.flatMap(root => root.stacks).find(stack => stack.effective > 0 && stack.status === 'ready');
  const within = delta >= -options.settings.placement!.maxSuccessLoss;
  const measured = { previous: rate(previousArm), refit: rate(refitArm), roots: development.length, delta,
    interval: comparison.ci, underpowered: comparison.underpowered };
  if (!within || !exercised)
    return { ...base, release: refit.digest, development: measured, activated: false, negativeTransfer: delta < 0,
      reason: !exercised ? 'no development call ran with the refit stack' : `development success fell by ${(-delta).toFixed(3)}, beyond the envelope` };
  activateRelease(options.root, refit.digest, { backend: { kind: options.settings.backend, loaded: [...refit.artifacts.map(item => item.artifact),
    ...refit.families.map(item => item.artifact)], identity: options.student.base },
  readiness: { definition: development[0]!.function, ok: true, effectiveIdentity: exercised.identity,
    ms: refitArm.roots.reduce((sum, root) => sum + root.ms, 0) / refitArm.roots.length } }, readActive(options.root)?.release ?? null);
  return { ...base, release: refit.digest, development: measured, activated: true, negativeTransfer: delta < 0 };
}

async function measureTransfer(program: FamilyProgram, family: PublishedFamily, kind: TransferOutcome['kind'], store: string,
  log: (line: string) => void): Promise<TransferOutcome> {
  const { options, data, release, releaseDir } = program;
  const base = { family: family.family, version: family.version, program: options.program.id, kind };
  const development = data.examples.filter(example => example.partition === 'development');
  if (!development.length) return { ...base, reason: 'no development roots' };
  const { members: _m, base: _b, ...version } = family;
  const { digest: _digest, ...current } = release;
  // The current release unchanged, plus the family at gate 1 on every binding: nothing is refit.
  const composed = sealManifest({ ...current, families: [...current.families.filter(item => item.family !== family.family), version],
    memberships: [...current.memberships.filter(item => item.family !== family.family),
      ...release.bindings.map(binding => ({ binding: binding.id, family: family.family, version: family.version, gate: 1 }))],
    receipts: { ...current.receipts, variant: `transfer:${family.family}@${family.version}` } } as Omit<WeightManifest, 'digest'>);
  storeRelease(options.root, composed);
  const directory = join(tuneDirectory(options.root), 'releases', composed.digest);
  const copy = (from: string, file: string) => {
    if (!existsSync(join(directory, file)) && existsSync(join(from, file))) {
      mkdirSync(dirname(join(directory, file)), { recursive: true }); copyFileSync(join(from, file), join(directory, file));
    }
  };
  copyNativeReleaseClosure([releaseDir, store], directory,
    [...composed.artifacts.map(item => item.artifact), ...composed.softStates.map(item => item.state), version.artifact]);
  for (const artifact of release.artifacts) if (artifact.exports?.gguf) copy(releaseDir, artifact.exports.gguf.file);
  if (version.exports?.gguf) copy(store, version.exports.gguf.file);
  log(`natlang tune: transfer of family ${family.family}@${family.version} to ${kind} program ${options.program.id}`);
  const currentArm = await evaluateRelease(options, data, release, development, 'hierarchy', 'development');
  const familyArm = await evaluateRelease(options, data, composed, development, `transfer:${family.family}`, 'development');
  saveArm(options.root, currentArm); saveArm(options.root, familyArm);
  const rate = (arm: ArmResult) => arm.roots.reduce((sum, root) => sum + rootSuccess(root), 0) / Math.max(arm.roots.length, 1);
  const comparison = pairedComparison(currentArm, familyArm);
  return { ...base, release: composed.digest, development: { current: rate(currentArm), withFamily: rate(familyArm), roots: development.length,
    delta: comparison.effect, interval: comparison.ci, underpowered: comparison.underpowered } };
}

/** Tuned programs of one base (plan §5.5 "run after newly tuned program batches"): every activated program release
 * is recorded; a family run resets the count of releases since the last one. */
export type FamilyRegistry = { readonly schema: 'natlang.tune-family-programs/v1';
  readonly programs: Readonly<Record<string, { readonly release: string; readonly at: string }>>;
  readonly sinceLastRun: readonly string[]; readonly lastRun: string | null };
const registryPath = (store: string) => join(store, 'programs.json');
export function familyRegistry(store: string): FamilyRegistry {
  return existsSync(registryPath(store)) ? JSON.parse(readFileSync(registryPath(store), 'utf8'))
    : { schema: 'natlang.tune-family-programs/v1', programs: {}, sinceLastRun: [], lastRun: null };
}
/** Record a newly activated program release; returns whether a family run is due (`every` new releases). */
export function registerTunedProgram(store: string, root: string, release: string, every: number | undefined): boolean {
  const registry = familyRegistry(store);
  if (registry.programs[root]?.release === release) return !!every && registry.sinceLastRun.length >= every;
  mkdirSync(store, { recursive: true });
  const next: FamilyRegistry = { ...registry, programs: { ...registry.programs, [root]: { release, at: new Date().toISOString() } },
    sinceLastRun: [...registry.sinceLastRun, root] };
  writeJson(registryPath(store), next);
  return !!every && next.sinceLastRun.length >= every;
}
function markFamiliesRun(store: string, roots: readonly string[]): void {
  const registry = familyRegistry(store);
  writeJson(registryPath(store), { ...registry, sinceLastRun: registry.sinceLastRun.filter(root => !roots.includes(root)), lastRun: new Date().toISOString() });
}
