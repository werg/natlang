/**
 * Skill-authoring episodes, `natlang.skill-episode/1` (S2 §4).
 *
 * An episode gives an author (teacher or student improver) a target program, a starting skill library and support
 * evidence. Query cases (and optional transfer cases from a related family) are sealed: the author sees only a host
 * evaluation ticket naming how many there are. The host evaluates starting and revised contexts on them afterwards.
 */
import { hexDigest } from '../native/hash.js';
import { splitFrontmatter } from './skill.js';

export const SKILL_EPISODE_SCHEMA = 'natlang.skill-episode/1';
export const AUTHORING_OPERATIONS = ['create', 'revise', 'select', 'repair-missing', 'repair-irrelevant', 'repair-incorrect',
  'retire', 'test'] as const;
export type AuthoringOperation = typeof AUTHORING_OPERATIONS[number];
export type DefectKind = 'missing' | 'irrelevant' | 'incorrect';

export type EpisodeCase = {
  id: string;
  /** Source group of the case; splits and leakage are decided by group, not instance. */
  group: string;
  args?: unknown[];
  folder?: Record<string, string>;
  expected?: unknown;
  expectedFiles?: Record<string, string>;
};

/** A starting program: an improvement-case program (files plus contract) or a `natlang.program/2` record. */
export type EpisodeTarget = {
  kind: 'improvement-case' | 'program';
  files: Record<string, string>;
  entry: string;
  exportName?: string;
  source: { schema: string; id: string };
};

/** Skill folders by name, each a map from path (relative to the skill folder) to contents. */
export type SkillFolders = Record<string, Record<string, string>>;

export type SkillEpisode = {
  version: typeof SKILL_EPISODE_SCHEMA;
  id: string;
  family: string;
  split: 'train' | 'validation' | 'test';
  source_groups: string[];
  license: string;
  target: EpisodeTarget;
  library: { kind: 'empty' | 'existing' | 'corrupted'; skills: SkillFolders;
    defect?: { kind: DefectKind; skill: string; detail: string; verified: boolean } };
  support: { cases: EpisodeCase[] };
  query: { cases: EpisodeCase[] };
  transfer?: { family: string; target: EpisodeTarget; cases: EpisodeCase[] };
  operations: AuthoringOperation[];
  limits: { maxSteps: number };
  provenance: Record<string, unknown>;
};

export type EvaluationTicket = { id: string; episode: string; query_cases: number; transfer_cases: number };
/** Author input omits sealed cases and host-only provenance, source identities, and repair labels. */
export type AuthorView = Omit<SkillEpisode, 'query' | 'transfer' | 'source_groups' | 'provenance' | 'library' | 'target'> & {
  /** Starting skills only. Repair labels/details remain on the host-side episode. */
  library: { kind: 'empty' | 'existing'; skills: SkillFolders };
  /** Source identity is evaluator metadata, not author input. */
  target: Omit<EpisodeTarget, 'source'> & { source: { schema: 'redacted'; id: 'redacted' } };
  evaluation: EvaluationTicket;
};
export type EpisodeDiagnostic = { path: string; code: string; message: string };

function canonical(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === 'object')
    return Object.fromEntries(Object.entries(value as Record<string, unknown>).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)
      .map(([key, item]) => [key, canonical(item)]));
  return value;
}
const canonicalJson = (value: unknown) => JSON.stringify(canonical(value));

/** The host-issued ticket for an episode's sealed cases. It commits to their content without revealing it. */
export function evaluationTicket(episode: SkillEpisode): EvaluationTicket {
  const sealed = canonicalJson({ id: episode.id, query: episode.query, transfer: episode.transfer ?? null });
  return { id: 'et1_' + hexDigest(sealed).slice(0, 32), episode: episode.id, query_cases: episode.query.cases.length,
    transfer_cases: episode.transfer?.cases.length ?? 0 };
}

export function authorView(episode: SkillEpisode): AuthorView {
  const { query: _query, transfer: _transfer, source_groups: _groups, provenance: _provenance,
    library, target, ...visible } = episode;
  return { ...structuredClone(visible),
    library: { kind: library.kind === 'empty' ? 'empty' : 'existing', skills: structuredClone(library.skills) },
    target: { ...structuredClone(target), source: { schema: 'redacted', id: 'redacted' } },
    evaluation: evaluationTicket(episode) };
}

/** Host side: confirm that a ticket was issued for exactly this episode's sealed cases. */
export function ticketMatches(episode: SkillEpisode, ticket: EvaluationTicket): boolean {
  return evaluationTicket(episode).id === ticket.id;
}

const strings = (value: unknown, out: string[] = []): string[] => {
  if (typeof value === 'string') out.push(value);
  else if (Array.isArray(value)) for (const item of value) strings(item, out);
  else if (value && typeof value === 'object') for (const item of Object.values(value)) strings(item, out);
  return out;
};

/**
 * Fragments of a sealed case's answer that would reveal it if they appeared in author-visible material: its expected
 * value as JSON, changed expected files, and the string leaves of both (including leaves of JSON files), leaving out
 * anything already present in the case's own inputs.
 */
export function answerFragments(item: EpisodeCase, minLength = 16, minLeaf = 8): string[] {
  const inputText = JSON.stringify([item.args ?? [], item.folder ?? {}]);
  const fresh = (text: string, min: number) => text.length >= min && !inputText.includes(JSON.stringify(text).slice(1, -1));
  const out = new Set<string>();
  const leaves = (value: unknown) => { for (const text of strings(value)) if (fresh(text, minLeaf)) out.add(text); };
  if (item.expected !== undefined) {
    const json = JSON.stringify(item.expected);
    if (fresh(json, minLength)) out.add(json);
    leaves(item.expected);
  }
  for (const [path, text] of Object.entries(item.expectedFiles ?? {})) {
    if (item.folder?.[path] === text) continue;
    if (fresh(text, minLength)) out.add(text);
    try { leaves(JSON.parse(text)); } catch { /* not JSON: the whole text is the fragment */ }
  }
  return [...out];
}

/** Validate shape and the leakage rules of S2 §4.3. An empty list means the episode is admissible. */
export function validateEpisode(episode: unknown): EpisodeDiagnostic[] {
  const out: EpisodeDiagnostic[] = [];
  const add = (path: string, code: string, message: string) => out.push({ path, code, message });
  const record = episode as Partial<SkillEpisode> | null;
  if (!record || typeof record !== 'object') return [{ path: '', code: 'episode-shape', message: 'not an object' }];
  if (record.version !== SKILL_EPISODE_SCHEMA) add('version', 'episode-version', `expected ${SKILL_EPISODE_SCHEMA}`);
  for (const key of ['id', 'family', 'license'] as const)
    if (typeof record[key] !== 'string' || !record[key]) add(key, 'episode-field', `${key} must be a non-empty string`);
  if (!['train', 'validation', 'test'].includes(record.split as string)) add('split', 'episode-split', 'split must be train, validation or test');
  if (!Array.isArray(record.source_groups) || !record.source_groups.length) add('source_groups', 'episode-field', 'source_groups must be non-empty');
  const target = (where: string, value: EpisodeTarget | undefined) => {
    if (!value || typeof value !== 'object') { add(where, 'episode-target', 'missing target'); return; }
    if (!['improvement-case', 'program'].includes(value.kind)) add(`${where}.kind`, 'episode-target', 'kind must be improvement-case or program');
    if (!value.files || typeof value.files[value.entry] !== 'string') add(`${where}.entry`, 'episode-target', 'entry must name a file of the target');
    if (!value.source?.id || !value.source?.schema) add(`${where}.source`, 'episode-target', 'target needs its source schema and id');
  };
  target('target', record.target);
  const cases = (where: string, value: { cases?: EpisodeCase[] } | undefined, needExpected: boolean): EpisodeCase[] => {
    if (!value || !Array.isArray(value.cases) || !value.cases.length) { add(where, 'episode-cases', `${where} needs at least one case`); return []; }
    value.cases.forEach((item, index) => {
      if (typeof item?.id !== 'string' || typeof item?.group !== 'string' || !item.group)
        add(`${where}.cases[${index}]`, 'episode-case', 'each case needs an id and a source group');
      if (needExpected && item.expected === undefined && item.expectedFiles === undefined)
        add(`${where}.cases[${index}]`, 'episode-case', 'sealed cases need expected results or files');
    });
    return value.cases;
  };
  const support = cases('support', record.support, false);
  const query = cases('query', record.query, true);
  const transfer = record.transfer ? cases('transfer', record.transfer, true) : [];
  if (record.transfer) {
    target('transfer.target', record.transfer.target);
    if (record.transfer.family === record.family) add('transfer.family', 'episode-transfer', 'transfer cases must come from a related, different family');
  }
  const ids = new Set<string>();
  for (const item of [...support, ...query, ...transfer]) {
    if (ids.has(item.id)) add(`cases.${item.id}`, 'episode-duplicate-case', `case id ${item.id} appears more than once`);
    ids.add(item.id);
  }
  const groups = (list: EpisodeCase[]) => new Set(list.map(item => item.group));
  const supportGroups = groups(support), queryGroups = groups(query), transferGroups = groups(transfer);
  for (const [name, sealed] of [['query', queryGroups], ['transfer', transferGroups]] as const)
    for (const group of sealed) if (supportGroups.has(group)) add(name, 'leak-group', `source group ${group} is in both support and ${name}`);
  for (const group of transferGroups) if (queryGroups.has(group)) add('transfer', 'leak-group', `source group ${group} is in both query and transfer`);
  if (!Array.isArray(record.operations) || !record.operations.length ||
    record.operations.some(op => !(AUTHORING_OPERATIONS as readonly string[]).includes(op)))
    add('operations', 'episode-operations', `operations must be a non-empty subset of ${AUTHORING_OPERATIONS.join(', ')}`);
  if (!record.limits || !Number.isSafeInteger(record.limits.maxSteps) || record.limits.maxSteps <= 0)
    add('limits.maxSteps', 'episode-limits', 'a positive integer step limit is required (the improver loop uses a TypeScript predicate)');
  const library = record.library;
  if (!library || !['empty', 'existing', 'corrupted'].includes(library.kind) || typeof library.skills !== 'object')
    add('library', 'episode-library', 'library needs kind (empty, existing, corrupted) and skills');
  else {
    if (library.kind === 'empty' && Object.keys(library.skills).length) add('library', 'episode-library', 'an empty library has no skills');
    if (library.kind === 'corrupted' && !library.defect) add('library.defect', 'episode-library', 'a corrupted library records its defect');
    if (library.defect && !['missing', 'irrelevant', 'incorrect'].includes(library.defect.kind))
      add('library.defect.kind', 'episode-library', 'defect kind must be missing, irrelevant or incorrect');
    for (const [name, files] of Object.entries(library.skills)) {
      const main = files['SKILL.md'];
      if (typeof main !== 'string') continue;
      let provenance: Record<string, unknown> | undefined;
      try {
        const data = splitFrontmatter(main)?.data as Record<string, unknown> | undefined;
        const block = (data?.natlang ?? (data?.metadata as Record<string, unknown> | undefined)?.natlang) as Record<string, unknown> | undefined;
        provenance = block?.provenance as Record<string, unknown> | undefined;
      } catch { provenance = undefined; }
      const madeFrom = Array.isArray(provenance?.support_groups) ? provenance!.support_groups as string[] : [];
      for (const group of madeFrom) if (queryGroups.has(group) || transferGroups.has(group))
        add(`library.skills.${name}`, 'leak-skill-provenance', `skill ${name} was authored from group ${group}, which is sealed in this episode`);
    }
  }
  // A sealed answer must not be visible anywhere the author can read. Fragments that support cases also produce (label
  // vocabularies, shared output templates) belong to the family's answer space, not to one sealed instance.
  const visible = JSON.stringify({ support: record.support, library: record.library?.skills ?? {}, target: record.target });
  const shared = new Set(support.flatMap(item => answerFragments(item)));
  for (const [name, list] of [['query', query], ['transfer', transfer]] as const)
    for (const item of list) {
      const leaked = answerFragments(item).find(fragment => !shared.has(fragment) && visible.includes(JSON.stringify(fragment).slice(1, -1)));
      if (leaked !== undefined) add(`${name}.${item.id}`, 'leak-answer',
        `an expected result of sealed case ${item.id} appears in author-visible material: ${JSON.stringify(leaked.slice(0, 60))}`);
    }
  return out;
}
