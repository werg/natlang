/**
 * The program inventor (DECOMPOSITION.md): one attempt turns the atlas of existing ideas into a new idea, checks that it
 * is new, has it built into a program with literal instances and an exact checker, qualifies it, has a teacher run it,
 * and records it so every later attempt sees it in the atlas.
 *
 * Judgments are `.nl` functions on frontier models (inventIdea, isNew, meetsBar, adjudicate). Exact checks, packaging,
 * replay and the coding agent that writes files are services: they touch the repository, the runtime and the outside
 * world.
 */
import { decide, nl } from '@natlang/node';
import { semanticRepair } from '@natlang/node/std';
import inventIdea from './inventIdea.nl';
import isNew from './isNew.nl';
import meetsBar from './meetsBar.nl';
import adjudicate from './adjudicate.nl';
import type { Atlas, AtlasEntry, AttemptResult, Disagreement, IdeaCard, LiveRun, ProgramView, Qualification, RejectedIdea, Requirement } from './types.js';

/** What the inventor needs from the outside world. */
export type InventorServices = {
  /** Rebuild the atlas from the registry, the repository and the accepted ledger (scripts/rl_tasks/program_atlas.py). */
  atlas(): Promise<Atlas>;
  /** The atlas entries nearest an idea (TF-IDF), plus every accepted idea. */
  near(idea: IdeaCard): Promise<AtlasEntry[]>;
  /** A frontier coding agent builds the idea into a submission directory (BUILD_BRIEF.md); returns its path. */
  build(idea: IdeaCard): Promise<string>;
  /** The same agent revises its submission to fix the listed problems. */
  revise(submission: string, problems: string[]): Promise<string>;
  /** Crisp qualification (scripts/rl_tasks/sources/invented.py qualify). */
  qualify(submission: string): Promise<Qualification>;
  /** The program's files and first instance, for review. */
  view(submission: string, idea: IdeaCard, metrics: string): Promise<ProgramView>;
  /** A teacher model runs every instance through the collector; one row per instance. */
  liveRun(submission: string): Promise<LiveRun[]>;
  /** Append an accepted idea to the ledger the atlas reads. */
  record(idea: IdeaCard, submission: string): Promise<void>;
  requirements: Requirement[];
};

/** Exact checks of an idea card before anything is built. */
export function ideaProblems(idea: IdeaCard, atlas: Atlas): string[] {
  const problems: string[] = [];
  if (!/^[a-z0-9]+(-[a-z0-9]+)*$/.test(idea.slug)) problems.push('slug is lowercase words joined by dashes');
  if (atlas.entries.some(entry => entry.id === `invented:${idea.slug}`)) problems.push(`slug ${idea.slug} is taken`);
  const questions = new Set(idea.judgments.map(judgment => judgment.question.trim().toLowerCase()));
  if (idea.judgments.length < 3 || questions.size < idea.judgments.length) problems.push('at least three judgments, each a different question');
  if (idea.constructs.length < 3) problems.push('at least three constructs');
  const known = new Set(atlas.entries.flatMap(entry => [entry.id, entry.name]));
  const unknown = idea.nearest.filter(name => !known.has(name));
  if (!idea.nearest.length || unknown.length) problems.push(`nearest names atlas entries as their lines start (unknown: ${unknown.join(', ') || 'none given'})`);
  return problems;
}

/** One attempt. `rejected` collects turned-down ideas so later attempts in the round go elsewhere. */
export async function attempt(services: InventorServices, rejected: RejectedIdea[]): Promise<AttemptResult> {
  const atlas = await services.atlas();
  const idea = await inventIdea(atlas.text, rejected);
  const problems = ideaProblems(idea, atlas);
  if (problems.length) {
    rejected.push({ slug: idea.slug, pitch: idea.pitch, reason: problems.join('; ') });
    return { slug: idea.slug, status: 'idea-problems', reason: problems.join('; ') };
  }
  const nearest = await services.near(idea);
  const novelty = await decide(isNew, idea, nearest);
  if (!novelty.value) {
    const reason: string = await nl`Name the entry of nearest that idea repeats and say in one sentence what is the same.`(idea, nearest);
    rejected.push({ slug: idea.slug, pitch: idea.pitch, reason });
    return { slug: idea.slug, status: 'not-new', reason };
  }

  // Build, then repair against the crisp qualification until it passes.
  const built = await semanticRepair(await services.build(idea), {
    check: async (submission: string) => (await services.qualify(submission)).problems,
    fix: (submission: string, found: string[]) => services.revise(submission, found),
    attempts: 3,
  });
  if (!built.ok) return { slug: idea.slug, status: 'build-failed', reason: built.problems.slice(0, 8).join('; '), submission: built.value };
  const submission = built.value;

  // The judged requirements, each by a model of another family than the inventor.
  const qualification = await services.qualify(submission);
  const program = await services.view(submission, idea, qualification.metrics);
  const judged = services.requirements.filter(requirement => requirement.check === 'judged' && requirement.id !== 'N1');
  const verdicts = await Promise.all(judged.map(async requirement => ({ requirement, met: (await decide(meetsBar, requirement, program)).value })));
  const unmet = verdicts.filter(verdict => !verdict.met).map(verdict => verdict.requirement.id);
  if (unmet.length) return { slug: idea.slug, status: 'below-bar', reason: `judged unmet: ${unmet.join(', ')}`, submission };

  // A teacher runs the program; a disagreement is settled before anything is recorded.
  const runs = await services.liveRun(submission);
  if (!runs.some(run => run.passed)) {
    const failed = runs[0]!;
    const disagreement: Disagreement = { program: program.files, instance: failed.instance, expected: failed.expected, answer: failed.answer, issues: failed.issues };
    const ruling = await decide(adjudicate, disagreement);
    return { slug: idea.slug, status: 'unsolvable', reason: `no instance passed live; adjudication: ${ruling.value}`, submission };
  }
  await services.record(idea, submission);
  return { slug: idea.slug, status: 'accepted', reason: `${runs.filter(run => run.passed).length}/${runs.length} instances passed live`, submission };
}

/** A round of `count` attempts, one after another, so each sees the ideas accepted before it. */
export async function inventRound(services: InventorServices, count: number): Promise<AttemptResult[]> {
  const rejected: RejectedIdea[] = [];
  const results: AttemptResult[] = [];
  for (let index = 0; index < count; index++) results.push(await attempt(services, rejected));
  return results;
}
