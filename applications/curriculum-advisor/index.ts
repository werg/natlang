/**
 * Advisory curriculum judgments (plans/NATLANG_NATIVE_REVIEW.md, P6). Two natural-language functions:
 * judgeAnswerEquivalence.nl decides whether a held extractive-answer row deserves a human look, and
 * nextCollectionBatch.nl proposes what the next collection batch should cover. This file is the crisp side: the cards and
 * facts the functions read, the checks of their answers, and the release of rows to human review.
 *
 * Nothing here admits a row or schedules a collection. `runtimeFailureReason` keeps holding the row whatever the
 * judgment says; a release only puts it in front of a person, and `admit` stays false.
 */
import { untrusted, type NatlangRuntime } from '@natlang/node';
import judgeAnswerEquivalence from './judgeAnswerEquivalence.nl';
import nextCollectionBatch from './nextCollectionBatch.nl';
import type { BatchProposal, CountRow, CoverageFacts, EquivalenceCard, EquivalenceJudgment, UntrustedEquivalenceCard } from './types.js';

export type * from './types.js';

/** Sources whose annotations are extractive spans (the sources `unreviewed_extractive_answer_equivalence` holds). */
export const EXTRACTIVE_SOURCES: readonly string[] = ['qasper', 'musique'];

const texts = (value: unknown): string[] => value === undefined || value === null ? [] :
  Array.isArray(value) ? value.flatMap(texts) : typeof value === 'string' ? [value] : [JSON.stringify(value)];

/** The card of a rejected extractive row: its question, the annotated answers and the model's answer. Undefined for other rows. */
export function equivalenceCard(row: { task?: Record<string, any>; outcome?: Record<string, any> }): EquivalenceCard | undefined {
  const record = row.task?.program_ir as Record<string, any> | undefined;
  if (!record || !EXTRACTIVE_SOURCES.includes(String(record.source)) || row.outcome?.accepted !== false) return undefined;
  if (!(row.outcome.rejection_reasons as string[] | undefined)?.includes('answer')) return undefined;
  const question = record.semantics?.inputs?.question ?? record.semantics?.inputs?.query ?? record.curriculum?.question ?? '';
  return { id: String(record.id), source: String(record.source), family: String(record.curriculum?.family ?? record.family ?? 'unknown'),
    question: String(typeof question === 'string' ? question : JSON.stringify(question)), gold: texts(record.semantics?.expected),
    answer: texts(row.outcome.value).join('\n') };
}

/** Problems of a judgment against its card: a quote must occur in the gold answers or the answer. */
export function checkEquivalenceJudgment(card: EquivalenceCard, out: EquivalenceJudgment): string[] {
  const sources = [...card.gold.map(String), String(card.answer)];
  return out.evidence.filter(({ quote }) => !sources.some(text => text.includes(quote)))
    .map(({ quote }) => `the quote ${JSON.stringify(quote.slice(0, 80))} occurs in neither the gold answers nor the answer`);
}

export type Release = {
  id: string,
  /** human_review: put the row in front of a person. keep_held: nothing changes. */
  release: 'human_review' | 'keep_held',
  verdict: EquivalenceJudgment['verdict'] | 'unjudged',
  reason: string,
  problems: string[],
  /** Constants: a release never admits a row, and the training admission is unchanged. */
  admit: false,
  training_admission: 'unchanged',
};

/** Only a checked judgment of equivalence releases a row, and only to a person. */
export function releaseToReview(card: EquivalenceCard, judgment: EquivalenceJudgment | undefined): Release {
  const problems = judgment ? checkEquivalenceJudgment(card, judgment) : ['no judgment'];
  const released = judgment?.verdict === 'equivalent' && problems.length === 0;
  return { id: card.id, release: released ? 'human_review' : 'keep_held', verdict: judgment?.verdict ?? 'unjudged',
    reason: judgment?.reason ?? '', problems, admit: false, training_admission: 'unchanged' };
}

export async function judgeEquivalences(runtime: NatlangRuntime, cards: EquivalenceCard[]): Promise<Release[]> {
  const releases: Release[] = [];
  for (const card of cards) {
    const wrapped: UntrustedEquivalenceCard = { ...card, question: untrusted(card.question, 'dataset question'),
      gold: card.gold.map(text => untrusted(text, 'dataset annotation')), answer: untrusted(card.answer, 'model answer') };
    let judgment: EquivalenceJudgment | undefined;
    try { judgment = await runtime.run(() => judgeAnswerEquivalence(wrapped), { name: 'judge-answer-equivalence' }) as EquivalenceJudgment; }
    catch { judgment = undefined; }
    releases.push(releaseToReview(card, judgment));
  }
  return releases;
}

// ---------------------------------------------------------------- next collection batch
const rows = (names: readonly string[], targets: Record<string, number>, counts: Map<string, number>, total: number): CountRow[] =>
  names.map(name => ({ name, target: targets[name] ?? 0, count: counts.get(name) ?? 0, share: total ? (counts.get(name) ?? 0) / total : 0 }));

/** Where the admitted cases stand against the target shares (the targets are data from curriculum.ts, passed in). */
export function coverageFacts(cases: { slice: string, domain: string }[], targets: { slices: Record<string, number>, domains: Record<string, number> },
    batchSize: number): CoverageFacts {
  const bySlice = new Map<string, number>(), byDomain = new Map<string, number>(), cells = new Map<string, number>();
  for (const item of cases) {
    bySlice.set(item.slice, (bySlice.get(item.slice) ?? 0) + 1);
    byDomain.set(item.domain, (byDomain.get(item.domain) ?? 0) + 1);
    cells.set(`${item.slice}\u0000${item.domain}`, (cells.get(`${item.slice}\u0000${item.domain}`) ?? 0) + 1);
  }
  const slices = Object.keys(targets.slices), domains = Object.keys(targets.domains);
  return { total: cases.length, batch_size: batchSize, slices: rows(slices, targets.slices, bySlice, cases.length),
    domains: rows(domains, targets.domains, byDomain, cases.length),
    cells: slices.flatMap(slice => domains.map(domain => ({ slice, domain, count: cells.get(`${slice}\u0000${domain}`) ?? 0 }))) };
}

/** Problems of a proposal against the coverage: known names, none with a zero target, a total within the batch. */
export function checkBatchProposal(coverage: CoverageFacts, out: BatchProposal): string[] {
  const problems: string[] = [];
  const slices = new Map(coverage.slices.map(item => [item.name, item.target])), domains = new Map(coverage.domains.map(item => [item.name, item.target]));
  let total = 0;
  for (const line of out.batch) {
    total += line.count;
    if (!slices.has(line.slice)) problems.push(`slice ${JSON.stringify(line.slice)} is not one of the slices`);
    else if (slices.get(line.slice) === 0) problems.push(`slice ${line.slice} has a target share of 0`);
    if (!domains.has(line.domain)) problems.push(`domain ${JSON.stringify(line.domain)} is not one of the domains`);
    else if (domains.get(line.domain) === 0) problems.push(`domain ${line.domain} has a target share of 0`);
  }
  if (total > coverage.batch_size) problems.push(`the counts add up to ${total}, above the batch size ${coverage.batch_size}`);
  return problems;
}

/** The proposal and its problems. A proposal with problems is still only advice, and the file says so. */
export async function proposeBatch(runtime: NatlangRuntime, coverage: CoverageFacts): Promise<{ value: BatchProposal, problems: string[] }> {
  const value = await runtime.run(() => nextCollectionBatch(coverage), { name: 'next-collection-batch' }) as BatchProposal;
  return { value, problems: checkBatchProposal(coverage, value) };
}
