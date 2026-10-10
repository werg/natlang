/**
 * The GEPA selection math, once. Two callers share it: the component-search engine (src/optimization) and the
 * program-improver application (applications/program-improver), which reaches it as the platform module `natlang:gepa`
 * (registered in runtime/node.ts, declared in compiler/project.ts). Callable folders may import only their siblings and
 * packages (runtime/modules.ts), so the application cannot import this file; a platform module is the sound way to give
 * it the same functions rather than a copy.
 *
 * Everything here is pure: members are plain data, draws are explicit (a seed in, a value and the next seed out), and
 * nothing reads a clock, a model or a store. Natural-language policies choose among these results; they never replace them.
 */

/** A member's score on one evaluation case. */
export type Score = { caseId: string; quality: number };
/** What the selection math reads of a population member (a component-search candidate or a source). */
export type Member = { id: string; quality: number; scores: readonly Score[]; cost?: number; modelCalls?: number };
export type Objective = 'quality' | 'source-size' | 'model-calls';

const missing = -1;
const scoreOn = (member: Member, caseId: string): number => member.scores.find(score => score.caseId === caseId)?.quality ?? missing;

/** The case ids of a population, sorted. */
export function caseIds(members: readonly Member[]): string[] {
  return [...new Set(members.flatMap(member => member.scores.map(score => score.caseId)))].sort();
}

/**
 * The per-case frontier: a member is on it when it has the highest quality on at least one case (a missing result counts
 * as -1). The winners are listed sorted, a member once for every case it wins.
 */
export function frontier(members: readonly Member[]): string[] {
  const winners: string[] = [];
  for (const caseId of caseIds(members)) {
    const best = Math.max(...members.map(member => scoreOn(member, caseId)));
    for (const member of members) if (scoreOn(member, caseId) === best) winners.push(member.id);
  }
  return winners.sort();
}

/** The frontier as a set of ids. */
export function frontierSet(members: readonly Member[]): Set<string> { return new Set(frontier(members)); }

/** The cases each frontier member wins, by member id (the evidence for choosing a parent). */
export function wonCases(members: readonly Member[]): Record<string, string[]> {
  const won: Record<string, string[]> = {};
  for (const caseId of caseIds(members)) {
    const best = Math.max(...members.map(member => scoreOn(member, caseId)));
    for (const member of members) if (scoreOn(member, caseId) === best) (won[member.id] ??= []).push(caseId);
  }
  return won;
}

/** One xorshift32 step: a value in [0, 1) and the seed for the next draw. */
export function draw(seed: number): { value: number; next: number } {
  let x = seed >>> 0;
  x ^= x << 13; x ^= x >>> 17; x ^= x << 5;
  const next = x >>> 0;
  return { value: next / 4294967296, next };
}

/** The element a draw value selects. */
export function pick<T>(choices: readonly T[], value: number): T {
  if (!choices.length) throw new RangeError('nothing to choose from');
  return choices[Math.min(choices.length - 1, Math.floor(value * choices.length))]!;
}

/** The ids a parent may be drawn from: the frontier winners, or every member when no case has a score. */
export function parentChoices(members: readonly Member[]): string[] {
  const winners = frontier(members);
  return winners.length ? winners : members.map(member => member.id).sort();
}

/** The crisp parent draw: a seeded pick over the frontier winners (a member that wins more cases is drawn more often). */
export function drawParent(members: readonly Member[], seed: number): { id: string; next: number } {
  const { value, next } = draw(seed);
  return { id: pick(parentChoices(members), value), next };
}

/** How a candidate is compared with another: eligibility first, then quality, then one measure where lower is better. */
export type BetterRule<T> = { eligible?: (member: T) => boolean; measure?: (member: T) => number | null };

/**
 * Whether `left` is better than `right`: an eligible member beats an ineligible one; a higher quality beats a lower;
 * equal qualities fall to the measure (lower is better, a known value beats an unknown one); otherwise not better.
 */
export function better<T extends { quality: number | null }>(left: T, right: T, rule: BetterRule<T> = {}): boolean {
  const eligible = rule.eligible ?? (() => true);
  if (!eligible(left)) return false;
  if (!eligible(right)) return true;
  if (left.quality !== right.quality) return left.quality! > right.quality!;
  if (!rule.measure) return false;
  const a = rule.measure(left), b = rule.measure(right);
  return a !== null && (b === null || a < b);
}

/** The tie-break measure of an objective: source size for `source-size`, model requests for `model-calls`, none for `quality`. */
export function objectiveMeasure(objective: Objective = 'quality'): ((member: { cost?: number; modelCalls?: number }) => number | null) | undefined {
  if (objective === 'source-size') return member => member.cost ?? null;
  if (objective === 'model-calls') return member => member.modelCalls ?? null;
  return undefined;
}

/** The members an incumbent may be chosen from: the highest quality, narrowed by the objective's measure. Sorted ids. */
export function leaders(members: readonly Member[], objective: Objective = 'quality'): string[] {
  const maximum = Math.max(...members.map(member => member.quality));
  let eligible = members.filter(member => member.quality === maximum);
  const measure = objectiveMeasure(objective);
  if (measure) {
    const minimum = Math.min(...eligible.map(member => measure(member) ?? Infinity));
    eligible = eligible.filter(member => (measure(member) ?? Infinity) === minimum);
  }
  return eligible.map(member => member.id).sort();
}

/** The crisp incumbent: the current one while it remains a leader, otherwise the leader with the smallest id. */
export function chooseIncumbent(members: readonly Member[], incumbent: string, objective: Objective = 'quality'): string {
  const chosen = leaders(members, objective);
  return chosen.includes(incumbent) ? incumbent : chosen[0]!;
}

/**
 * The members that stay when a population is cut to `limit`: protected members always, then frontier members, then higher
 * quality, then smaller id. The kept members keep their order in `members`; duplicates by id are dropped (first wins).
 */
export function prune<T extends Member>(members: readonly T[], protectedIds: Iterable<string>, limit: number): T[] {
  if (!Number.isSafeInteger(limit) || limit < 1) throw new RangeError('population limit must be a positive integer');
  const keep = new Set(protectedIds);
  const unique = members.filter((member, index) => members.findIndex(other => other.id === member.id) === index);
  const onFrontier = frontierSet(unique);
  const kept = new Set(unique.filter(member => keep.has(member.id)).map(member => member.id));
  const rest = unique.filter(member => !kept.has(member.id))
    .sort((a, b) => Number(onFrontier.has(b.id)) - Number(onFrontier.has(a.id)) || b.quality - a.quality || a.id.localeCompare(b.id));
  for (const member of rest) { if (kept.size >= limit) break; kept.add(member.id); }
  return unique.filter(member => kept.has(member.id));
}

const isRecord = (value: unknown): value is Record<string, unknown> => value !== null && typeof value === 'object' && !Array.isArray(value);
const canonicalJson = (value: unknown): string => JSON.stringify(value, (_key, item) =>
  isRecord(item) ? Object.fromEntries(Object.entries(item).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)) : item) ?? 'undefined';

/**
 * What is wrong with an edited component set, as text that says what to change, or '' when nothing is. The edit of a
 * component search changes only the selected keys of its parent and keeps every kind, every slot id list and the number
 * of segments of each instruction template; the segments are an array of strings. This is the exact check that stands
 * behind the natural-language component edit (componentSearchStep/rewriteComponents.ts sends the text back once; the
 * engine's `check` applies it again).
 */
export function componentProblem(parent: Record<string, unknown>, candidate: unknown, keys: readonly string[]): string {
  if (!isRecord(candidate)) return 'components.json holds one JSON object whose keys are the component keys.';
  const missing = Object.keys(parent).filter(key => !(key in candidate));
  if (missing.length) return `components.json keeps every component of the parent; these are missing: ${missing.join(', ')}.`;
  const unknown = Object.keys(candidate).filter(key => !(key in parent));
  if (unknown.length) return `components.json holds only the components of the parent; these are not among them: ${unknown.join(', ')}.`;
  for (const key of Object.keys(parent)) {
    const before = parent[key], after = candidate[key];
    if (!keys.includes(key)) {
      if (canonicalJson(before) !== canonicalJson(after)) return `The component ${key} is not selected, so it keeps its parent value; the selected components are: ${keys.join(', ')}.`;
      continue;
    }
    if (!isRecord(before) || !isRecord(after)) return `The component ${key} is an object with a kind.`;
    if (after.kind !== before.kind) return `The component ${key} keeps its kind \`${String(before.kind)}\`.`;
    if (before.kind === 'program.guidance') {
      if (typeof after.text !== 'string') return `The guidance component ${key} has its text as a string.`;
      continue;
    }
    const was = before.template, now = after.template;
    if (!isRecord(was) || !isRecord(now)) return `The instructions component ${key} has a template object.`;
    if (!Array.isArray(now.segments) || now.segments.some(segment => typeof segment !== 'string'))
      return `The segments of ${key} are a JSON array of strings; each static segment is one string in that array.`;
    if (canonicalJson(now.slotIds) !== canonicalJson(was.slotIds)) return `The slot ids of ${key} keep their parent value: ${canonicalJson(was.slotIds)}.`;
    if (now.segments.length !== (was.segments as unknown[]).length) return `${key} keeps its ${(was.segments as unknown[]).length} segments (it has ${now.segments.length}); write the new text inside the existing segments.`;
  }
  return '';
}

/** The module natlang code imports as `natlang:gepa`. */
export const gepaModule = { __esModule: true, caseIds, frontier, frontierSet, wonCases, draw, pick, parentChoices, drawParent, better,
  objectiveMeasure, leaders, chooseIncumbent, prune, componentProblem } as const;
