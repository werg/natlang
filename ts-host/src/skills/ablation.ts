import YAML from 'yaml';
import { splitFrontmatter } from './skill.js';

export type SkillFileValue = string | Uint8Array;
export type SkillFileMap = Readonly<Record<string, SkillFileValue>>;
export type SkillAblationKind = 'leave_one_skill_out' | 'baseline_description' | 'baseline_body';

/** A candidate owns a private byte snapshot. Each read returns detached values. */
export class SkillAblationCandidate {
  readonly kind: SkillAblationKind;
  readonly skillName: string;
  readonly changedPaths: readonly string[];
  readonly #snapshot: ReadonlyMap<string, SkillFileValue>;

  constructor(kind: SkillAblationKind, skillName: string, files: SkillFileMap, changedPaths: readonly string[]) {
    this.kind = kind;
    this.skillName = skillName;
    this.changedPaths = Object.freeze([...changedPaths].sort());
    this.#snapshot = new Map(Object.entries(files).map(([path, value]) => [path, cloneValue(value)]));
    Object.freeze(this);
  }

  get files(): SkillFileMap {
    const copy: Record<string, SkillFileValue> = Object.create(null);
    for (const [path, value] of this.#snapshot) copy[path] = cloneValue(value);
    return Object.freeze(copy);
  }
}

export type SkillAblationBuild = {
  candidates: readonly SkillAblationCandidate[];
  skipped: readonly { kind: SkillAblationKind; skillName: string; reason: 'unchanged' | 'missing_common_skill' | 'invalid_frontmatter' }[];
};

/**
 * Create isolated host-side skill variants. `entryRoot` is the relative path to the skills directory
 * (usually `skills`); executable files, including the selected root program, are otherwise untouched.
 */
export function buildSkillAblations(baseline: SkillFileMap, selected: SkillFileMap, entryRoot = 'skills'): SkillAblationBuild {
  const root = normalizeRoot(entryRoot);
  const baselineCopy = cloneMap(baseline);
  const selectedCopy = cloneMap(selected);
  const baselineNames = skillNames(baselineCopy, root);
  const selectedNames = skillNames(selectedCopy, root);
  const candidates: SkillAblationCandidate[] = [];
  const skipped: SkillAblationBuild['skipped'][number][] = [];

  for (const name of selectedNames) {
    const prefix = `${root}/${name}/`;
    const files = cloneMap(selectedCopy);
    const changed = Object.keys(files).filter(path => path.startsWith(prefix));
    for (const path of changed) delete (files as Record<string, SkillFileValue>)[path];
    candidates.push(new SkillAblationCandidate('leave_one_skill_out', name, files, changed));
  }

  for (const name of selectedNames) {
    if (!baselineNames.has(name)) {
      skipped.push({ kind: 'baseline_description', skillName: name, reason: 'missing_common_skill' },
        { kind: 'baseline_body', skillName: name, reason: 'missing_common_skill' });
      continue;
    }
    const path = `${root}/${name}/SKILL.md`;
    const oldText = asText(baselineCopy[path]);
    const newText = asText(selectedCopy[path]);
    if (oldText === undefined || newText === undefined) {
      skipped.push({ kind: 'baseline_description', skillName: name, reason: 'invalid_frontmatter' },
        { kind: 'baseline_body', skillName: name, reason: 'invalid_frontmatter' });
      continue;
    }
    const oldParts = splitFrontmatter(oldText);
    const newParts = splitFrontmatter(newText);
    if (!oldParts || !newParts || !isRecord(oldParts.data) || !isRecord(newParts.data)) {
      skipped.push({ kind: 'baseline_description', skillName: name, reason: 'invalid_frontmatter' },
        { kind: 'baseline_body', skillName: name, reason: 'invalid_frontmatter' });
      continue;
    }
    const oldData = oldParts.data;
    const newData = newParts.data;

    const metadataSame = ['description', 'summary'].every(key =>
      Object.hasOwn(oldData, key) === Object.hasOwn(newData, key) &&
      equivalent(oldData[key], newData[key]));
    if (metadataSame) skipped.push({ kind: 'baseline_description', skillName: name, reason: 'unchanged' });
    else {
      const restoredMetadata = { ...newData };
      for (const key of ['description', 'summary']) {
        if (Object.hasOwn(oldData, key)) restoredMetadata[key] = oldData[key];
        else delete restoredMetadata[key];
      }
      candidates.push(new SkillAblationCandidate('baseline_description', name,
        { ...selectedCopy, [path]: serializeSkill(restoredMetadata, newParts.body) }, [path]));
    }

    if (newParts.body === oldParts.body) skipped.push({ kind: 'baseline_body', skillName: name, reason: 'unchanged' });
    else candidates.push(new SkillAblationCandidate('baseline_body', name,
      { ...selectedCopy, [path]: serializeSkill(newData, oldParts.body) }, [path]));
  }
  return { candidates: Object.freeze(candidates), skipped: Object.freeze(skipped) };
}

export type SkillAblationOutcome = {
  baselinePassed?: boolean;
  ablatedPassed?: boolean;
};
export type SkillUseCounts = Readonly<Record<'offered' | 'body_read' | 'support_file_read' | 'helper_invoked', number>>;
export type SkillAblationEffect = {
  skillName: string;
  classification: 'unknown' | 'helpful-at-tested-context' | 'nonessential-at-tested-context';
  testedContexts: number;
  baselinePasses: number;
  ablatedPasses: number;
  pairedPasses: number;
  observedUse: SkillUseCounts;
  interpretation: 'host_observations_only_not_cognitive_use';
};

/** Summarize paired outcomes and host observations without inferring that reads imply comprehension. */
export function summarizeSkillAblation(skillName: string, outcomes: readonly SkillAblationOutcome[],
  events: readonly { kind?: string; phase?: string; skill_name?: string }[] = []): SkillAblationEffect {
  const counts = { offered: 0, body_read: 0, support_file_read: 0, helper_invoked: 0 };
  for (const event of events) if (event.kind === 'skill_use' && event.skill_name === skillName &&
    Object.hasOwn(counts, event.phase ?? '')) counts[event.phase as keyof typeof counts]++;
  let baselinePasses = 0, ablatedPasses = 0, pairedPasses = 0, pairedFailures = 0;
  for (const outcome of outcomes) {
    if (outcome.baselinePassed === true) baselinePasses++;
    if (outcome.ablatedPassed === true) ablatedPasses++;
    if (outcome.baselinePassed === true && outcome.ablatedPassed === true) pairedPasses++;
    if (outcome.baselinePassed === true && outcome.ablatedPassed === false) pairedFailures++;
  }
  const paired = outcomes.filter(outcome => typeof outcome.baselinePassed === 'boolean' && typeof outcome.ablatedPassed === 'boolean');
  const classification = paired.length === 0 ? 'unknown' : pairedFailures > 0 ? 'helpful-at-tested-context' :
    paired.every(outcome => outcome.baselinePassed === true && outcome.ablatedPassed === true)
      ? 'nonessential-at-tested-context' : 'unknown';
  return { skillName, classification, testedContexts: paired.length, baselinePasses, ablatedPasses, pairedPasses,
    observedUse: Object.freeze(counts), interpretation: 'host_observations_only_not_cognitive_use' };
}

function skillNames(files: SkillFileMap, root: string): Set<string> {
  const prefix = `${root}/`;
  return new Set(Object.keys(files).flatMap(path => {
    if (!path.startsWith(prefix)) return [];
    const rest = path.slice(prefix.length);
    const split = rest.indexOf('/');
    if (split <= 0 || rest.slice(split + 1) !== 'SKILL.md') return [];
    return [rest.slice(0, split)];
  }));
}
function normalizeRoot(root: string): string {
  if (root.startsWith('/') || root.startsWith('\\') || /^[A-Za-z]:/.test(root) || root.includes('\\'))
    throw new Error('entryRoot must be a safe relative path');
  const value = root.replace(/\/+$/g, '');
  if (!value || value.split('/').some(part => !/^[A-Za-z0-9._-]+$/.test(part) || part === '.' || part === '..'))
    throw new Error('entryRoot must be a safe relative path');
  return value;
}
function cloneValue(value: SkillFileValue): SkillFileValue { return typeof value === 'string' ? value : new Uint8Array(value); }
function cloneMap(files: SkillFileMap): Record<string, SkillFileValue> {
  const out: Record<string, SkillFileValue> = Object.create(null);
  for (const [path, value] of Object.entries(files)) {
    if (!path || path.startsWith('/') || path.split('/').some(part => !part || part === '..' || part === '.'))
      throw new Error(`unsafe skill snapshot path: ${path}`);
    out[path] = cloneValue(value);
  }
  return out;
}
function asText(value: SkillFileValue | undefined): string | undefined {
  if (typeof value === 'string') return value;
  if (value instanceof Uint8Array) return new TextDecoder('utf-8', { fatal: true }).decode(value);
  return undefined;
}
function isRecord(value: unknown): value is Record<string, unknown> { return typeof value === 'object' && value !== null && !Array.isArray(value); }
function equivalent(left: unknown, right: unknown): boolean {
  if (Object.is(left, right)) return true;
  if (Array.isArray(left) || Array.isArray(right))
    return Array.isArray(left) && Array.isArray(right) && left.length === right.length && left.every((value, index) => equivalent(value, right[index]));
  if (!isRecord(left) || !isRecord(right)) return false;
  const leftKeys = Object.keys(left).sort(), rightKeys = Object.keys(right).sort();
  return leftKeys.length === rightKeys.length && leftKeys.every((key, index) => key === rightKeys[index] && equivalent(left[key], right[key]));
}
function serializeSkill(frontmatter: Record<string, unknown>, body: string): string {
  return `---\n${YAML.stringify(frontmatter).trimEnd()}\n---\n${body}`;
}
