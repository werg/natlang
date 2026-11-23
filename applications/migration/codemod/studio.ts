/**
 * Codemods (WS6 §3 A9), part of the migration package: deterministic regex/literal codemod rules cover most files with
 * zero model calls; the semantic remainder runs through this package's migration loop (`migrate`: understand, survey,
 * classify, plan, edit, validate, triage and repair with the natural-language stopping judgment) over an in-memory
 * repository whose validation is the caller's exact `verify`; and repeated verified semantic edits are mined into new
 * crisp rules that are checked against already-migrated pairs before admission.
 */
import { untrusted, type NatlangRuntime } from '@natlang/node';
import { RepositoryMigration, migrate } from '../index.js';
import type { Check } from '../types.js';
import mineCrispRule from './mineCrispRule.nl';
import type {
  CodemodReport,
  CodemodSpec,
  CrispRule,
  FileResult,
  MinedRuleCandidate,
  MinedRuleOutcome,
  SourceFile,
  VerifiedMigrationPair,
} from './types.js';

export type * from './types.js';

/** Exact verifier callback (typecheck + tests): returns a list of problems, empty when the migrated file is green. */
export type VerifyFile = (
  path: string,
  content: string,
  workspaceFiles: ReadonlyMap<string, string>,
) => string[] | Promise<string[]>;

export type CodemodOptions = {
  /** The runtime the migration loop runs its natural-language stages on (required when a file needs the semantic stage). */
  runtime?: NatlangRuntime;
  /** Initial deterministic crisp codemod rules (including any previously mined rules). */
  crispRules?: readonly CrispRule[];
  /** Exact verifier (`tsc` + unit tests). */
  verify: VerifyFile;
  /** Repair rounds of the migration loop after its first candidate (default `2`). */
  attempts?: number;
  /** Minimum number of green semantic migrations before attempting to mine a reusable crisp rule (default `2`). */
  minPairsToMine?: number;
  /** Whether to run crisp rule mining after semantic migrations (default `true`). */
  mineRules?: boolean;
};

/** Helper to mark raw source text as `Untrusted<string>` at the repository boundary. */
export function createSourceFile(path: string, content: string): SourceFile {
  return { path, content: untrusted(content, `repo file ${path}`) };
}

/** Apply one deterministic crisp rule to source text; returns unchanged output if the rule regex is invalid. */
export function applyCrispRule(source: string, rule: CrispRule): { output: string; changed: boolean } {
  if (!rule.find) return { output: source, changed: false };
  if (rule.regex) {
    try {
      const flags = rule.flags && rule.flags.includes('g') ? rule.flags : `${rule.flags ?? ''}g`;
      const pattern = new RegExp(rule.find, flags);
      const output = source.replace(pattern, rule.replace);
      return { output, changed: output !== source };
    } catch {
      return { output: source, changed: false };
    }
  }
  if (!source.includes(rule.find)) return { output: source, changed: false };
  const output = source.split(rule.find).join(rule.replace);
  return { output, changed: output !== source };
}

/** Apply a sequence of deterministic crisp rules in order with zero LLM calls. */
export function applyCrispRules(
  source: string,
  rules: readonly CrispRule[],
): { output: string; ruleIds: string[]; changed: boolean } {
  let current = source;
  const ruleIds: string[] = [];
  for (const rule of rules) {
    const applied = applyCrispRule(current, rule);
    if (applied.changed) {
      current = applied.output;
      ruleIds.push(rule.id);
    }
  }
  return { output: current, ruleIds, changed: current !== source };
}

/**
 * Verify a candidate mined crisp rule against already-migrated pairs and clean baseline files before admitting it.
 * A rule is admitted only when it reproduces (or passes verification on) at least `minVerifiedPairs` semantic pairs
 * and introduces zero regressions on already-migrated outputs.
 */
export async function verifyMinedRule(
  candidate: MinedRuleCandidate,
  verifiedPairs: readonly VerifiedMigrationPair[],
  cleanBaseline: ReadonlyMap<string, string>,
  verify: VerifyFile,
  ruleId: string,
  minVerifiedPairs = 1,
): Promise<MinedRuleOutcome> {
  const rule: CrispRule = {
    id: ruleId,
    description: candidate.description,
    find: candidate.find,
    replace: candidate.replace,
    ...(candidate.regex !== undefined ? { regex: candidate.regex } : {}),
    ...(candidate.flags !== undefined ? { flags: candidate.flags } : {}),
    origin: 'mined',
  };

  if (!rule.find || rule.find === rule.replace) {
    return { rule, admitted: false, verifiedPairs: 0, regressions: 0, reason: 'Rule pattern is empty or a no-op.' };
  }

  if (rule.regex) {
    try {
      new RegExp(rule.find, rule.flags ?? 'g');
    } catch (error) {
      return {
        rule,
        admitted: false,
        verifiedPairs: 0,
        regressions: 0,
        reason: `Invalid regex: ${error instanceof Error ? error.message : String(error)}`,
      };
    }
  }

  let matchedCount = 0;
  let regressions = 0;
  for (const pair of verifiedPairs) {
    const applied = applyCrispRule(pair.before, rule);
    if (!applied.changed) continue;
    const trialWorkspace = new Map(cleanBaseline);
    trialWorkspace.set(pair.path, applied.output);
    const problems = await verify(pair.path, applied.output, trialWorkspace);
    if (applied.output === pair.after || problems.length === 0) {
      matchedCount++;
    } else {
      regressions++;
    }
  }

  // Ensure the mined rule is idempotent and does not break already-clean migrated files.
  for (const [path, cleanContent] of cleanBaseline.entries()) {
    const reapplied = applyCrispRule(cleanContent, rule);
    if (reapplied.changed) {
      const trialWorkspace = new Map(cleanBaseline);
      trialWorkspace.set(path, reapplied.output);
      const problems = await verify(path, reapplied.output, trialWorkspace);
      if (problems.length > 0) regressions++;
    }
  }

  const admitted = regressions === 0 && matchedCount >= minVerifiedPairs;
  return {
    rule,
    admitted,
    verifiedPairs: matchedCount,
    regressions,
    ...(admitted
      ? {}
      : {
          reason:
            regressions > 0
              ? `Mined rule caused ${regressions} regression(s) on verified files.`
              : `Mined rule verified on only ${matchedCount}/${minVerifiedPairs} required pair(s).`,
        }),
  };
}

/** The request the migration loop reads for a codemod spec: what to change and how. */
export function migrationRequest(spec: CodemodSpec): string {
  const examples = (spec.examples ?? []).map(example => `Before:\n${example.before}\nAfter:\n${example.after}`).join('\n\n');
  return [spec.title, spec.description, spec.guidance, examples && `Examples:\n${examples}`].filter(Boolean).join('\n\n');
}

/**
 * Migrate a repository:
 * 1. Crisp rules run first on every file (0 model calls on covered or already-clean files).
 * 2. Files the verifier still rejects run through the migration loop (`migrate`) as one in-memory repository whose
 *    validation is `verify` on each of them; the loop's last candidate is kept for the files it made green.
 * 3. Once repeated semantic migrations pass, `mineCrispRule.nl` proposes a reusable crisp rule, which is verified
 *    against already-migrated pairs before admission.
 */
export async function migrateRepository(
  files: readonly SourceFile[],
  spec: CodemodSpec,
  options: CodemodOptions,
): Promise<CodemodReport> {
  const activeRules: CrispRule[] = [...(options.crispRules ?? [])];
  const attempts = options.attempts ?? 2;
  const minPairsToMine = options.minPairsToMine ?? 2;
  const shouldMine = options.mineRules ?? true;

  const workingContents = new Map<string, string>();
  for (const file of files) workingContents.set(file.path, file.content);

  // Stage 1: Deterministic crisp pass (0 model calls).
  const crispDrafts = new Map<string, { output: string; ruleIds: string[]; changed: boolean }>();
  for (const file of files) {
    const draft = applyCrispRules(file.content, activeRules);
    crispDrafts.set(file.path, draft);
    workingContents.set(file.path, draft.output);
  }

  const results: FileResult[] = [];
  const remaining: { file: SourceFile; ruleIds: string[] }[] = [];
  for (const file of files) {
    const draft = crispDrafts.get(file.path)!;
    const problems = await options.verify(file.path, draft.output, workingContents);
    if (problems.length === 0) {
      results.push({ path: file.path, status: draft.changed ? 'crisp' : 'unchanged', original: file.content, migrated: draft.output,
        ruleIds: draft.ruleIds, attempts: 0, problems: [] });
    } else {
      remaining.push({ file, ruleIds: draft.ruleIds });
    }
  }

  // Stage 2: the migration loop over the semantic remainder, validated by the exact verifier.
  let llmCalls = 0;
  let migrationSummary: string | undefined;
  const semanticPairs: VerifiedMigrationPair[] = [];
  if (remaining.length) {
    if (!options.runtime) throw new Error('migrateRepository needs options.runtime to run the migration loop on the files crisp rules did not fix');
    const targets = remaining.map(item => item.file.path);
    const validator = async (contents: Readonly<Record<string, string>>): Promise<Check[]> => {
      const workspace = new Map(Object.entries(contents));
      return Promise.all(targets.map(async path => {
        const problems = await options.verify(path, contents[path]!, workspace);
        const output = problems.join('\n');
        return { id: path, status: problems.length ? 'failed' as const : 'passed' as const, output, output_bytes: output.length, truncated: false };
      }));
    };
    const repository = new RepositoryMigration('.', { files: [...workingContents.keys()], contents: Object.fromEntries(workingContents), validator });
    await repository.open();
    llmCalls++;
    const migration = await migrate(options.runtime, repository, migrationRequest(spec), { attempts, checks: targets });
    migrationSummary = migration.summary;
    const checks = new Map(migration.report.checks.map(check => [check.id, check]));
    for (const item of remaining) {
      const path = item.file.path;
      const migrated = repository.read(path, migration.report.revision);
      const check = checks.get(path);
      if (check?.status === 'passed') {
        workingContents.set(path, migrated);
        semanticPairs.push({ path, before: item.file.content, after: migrated });
        results.push({ path, status: 'semantic', original: item.file.content, migrated, ruleIds: item.ruleIds,
          attempts: migration.rounds + 1, problems: [] });
      } else {
        results.push({ path, status: 'failed', original: item.file.content, migrated, ruleIds: item.ruleIds, attempts: migration.rounds + 1,
          problems: check ? check.output.split('\n').filter(Boolean) : ['the migration loop reported no check for this file'] });
      }
    }
  }

  // Stage 3: Mine a reusable crisp rule from verified semantic migrations and verify before admission.
  const minedRules: MinedRuleOutcome[] = [];
  const admittedRules: CrispRule[] = [];

  if (shouldMine && semanticPairs.length >= minPairsToMine) {
    llmCalls++;
    const candidate = await mineCrispRule(spec, semanticPairs);
    const verified = await verifyMinedRule(
      candidate,
      semanticPairs,
      workingContents,
      options.verify,
      `mined-${spec.id}-${activeRules.length + 1}`,
      Math.min(minPairsToMine, semanticPairs.length),
    );
    minedRules.push(verified);
    if (verified.admitted) admittedRules.push(verified.rule);
  }

  const crispFiles = results.filter(r => r.status === 'crisp').length;
  const semanticFiles = results.filter(r => r.status === 'semantic').length;
  const unchangedFiles = results.filter(r => r.status === 'unchanged').length;
  const failedFiles = results.filter(r => r.status === 'failed').length;

  return {
    specId: spec.id,
    totalFiles: files.length,
    crispFiles,
    semanticFiles,
    unchangedFiles,
    failedFiles,
    llmCalls,
    ...(migrationSummary !== undefined ? { migrationSummary } : {}),
    results,
    admittedRules,
    minedRules,
  };
}
