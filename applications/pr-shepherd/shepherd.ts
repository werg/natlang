/**
 * PR Shepherd (Tier 1 Flagship A1, plans/semantic-fp/06_FLAGSHIP_APPLICATIONS.md §3 & §3.5):
 * - Crisp fast-path via `pluggable` (`isTrivialHunk`) skips lockfiles, snapshots, generated artifacts, whitespace-only
 *   diffs, and import-only reorders with zero model calls (targeting >= 60% crisp share on routine PR hunks).
 * - Non-trivial hunks are classified by finite-record `classifyHunk.nl` (`readout: decision`).
 * - Each team rule is bound onto `checkHunk.nl` via `withContext` (preserving its `ensures: quotesMatch` verbatim quote
 *   verifier), then fanned out concurrently with `semanticMap`.
 * - Findings are ordered by debiased pairwise `semanticRank`.
 * - Merge readiness runs through `escalate(mergeVerdict, { to: 'teacher', floor: 0.85, risk: { block: 0.25 } })`.
 * - Optional documentation fixes run through `proposeAndVerify` over a `Folder` transaction and only install when the
 *   exact verifier passes.
 */
import { pluggable, type Folder, type FolderHandle, type FolderProposal, type PluggableSetting } from '@natlang/node';
import { escalate, proposeAndVerify, semanticMap, semanticRank, withContext, type Problems } from '@natlang/node/std';
import checkHunk from './checkHunk.nl';
import classifyHunk from './classifyHunk.nl';
import isTrivialHunkNl from './isTrivialHunk.nl';
import mergeVerdict from './mergeVerdict.nl';
import proposeDocFix from './proposeDocFix.nl';
import type { DocFixDraft, DocFixResult, Finding, Hunk, HunkClass, HunkSummary, Rule, ShepherdReport } from './types.js';

export type ShepherdOptions = {
  /** Mode for the pluggable trivial-hunk fast-path (`'crisp'` default, `'nl'`, or `'shadow'`). */
  trivialMode?: PluggableSetting;
  /** Stronger model name for asymmetric escalation (default `'teacher'`). */
  escalateTo?: string;
  /** Confidence threshold below which `mergeVerdict` escalates (default `0.85`). */
  floor?: number;
  /** Asymmetric risk threshold for `'block'` probability (default `0.25`). */
  blockRiskFloor?: number;
  /** Maximum concurrent semantic calls (default `6`). */
  concurrency?: number;
  /** Optional writable `Folder` workspace for proposing a verified documentation fix. */
  docWorkspace?: Folder;
  /** Target documentation path inside `docWorkspace` (for example `'docs/SECURITY.md'`). */
  docTargetPath?: string;
  /** Maximum re-proposals for `proposeAndVerify` (default `2`). */
  docFixAttempts?: number;
  /** Custom exact verifier for doc-fix proposals. */
  verifyDocFix?: (proposal: FolderProposal<DocFixDraft>, findings: Finding[], targetPath: string) => Problems | Promise<Problems>;
};

const LOCKFILE_RE = /(?:^|\/)(?:package-lock\.json|yarn\.lock|pnpm-lock\.yaml|Cargo\.lock|poetry\.lock|uv\.lock|Gemfile\.lock|composer\.lock|go\.sum)$/;
const SNAPSHOT_RE = /(?:^|\/)__snapshots__\/|\.snap$/;
const GENERATED_RE = /\.min\.(?:js|css)$|\.generated\.[^/]+$|\.pb\.(?:ts|js|go)$/;
const IMPORT_LINE_RE = /^(?:import\s|export\s+(?:type\s+)?\{[^}]*\}\s+from\s|from\s+\S+\s+import\s)/;

/** Extract added and removed content lines from a unified diff hunk (ignoring diff headers). */
function diffChangedLines(patch: string): { added: string[]; removed: string[] } {
  const added: string[] = [];
  const removed: string[] = [];
  for (const line of patch.split(/\r?\n/)) {
    if (line.startsWith('+++') || line.startsWith('---') || line.startsWith('@@')) continue;
    if (line.startsWith('+')) added.push(line.slice(1));
    else if (line.startsWith('-')) removed.push(line.slice(1));
  }
  return { added, removed };
}

/** Deterministic fast-path recognizing routine trivial hunks with zero model calls. */
export function isTrivialHunkCrisp(hunk: Hunk): boolean {
  if (hunk.isGenerated) return true;
  if (LOCKFILE_RE.test(hunk.file) || SNAPSHOT_RE.test(hunk.file) || GENERATED_RE.test(hunk.file)) return true;
  const { added, removed } = diffChangedLines(String(hunk.patch));
  if (added.length === 0 && removed.length === 0) return true;
  // Whitespace-only formatting change: identical sequence once trimmed/collapsed.
  const normAdded = added.map(l => l.replace(/\s+/g, ' ').trim()).filter(Boolean);
  const normRemoved = removed.map(l => l.replace(/\s+/g, ' ').trim()).filter(Boolean);
  if (normAdded.length === normRemoved.length && normAdded.every((line, i) => line === normRemoved[i])) return true;
  // Import-order-only change: every non-empty changed line is an import and sorted multisets match.
  if (normAdded.length > 0 && normAdded.length === normRemoved.length &&
    normAdded.every(l => IMPORT_LINE_RE.test(l)) && normRemoved.every(l => IMPORT_LINE_RE.test(l))) {
    const sortedA = [...normAdded].sort();
    const sortedR = [...normRemoved].sort();
    if (sortedA.every((line, i) => line === sortedR[i])) return true;
  }
  return false;
}

/** Pluggable trivial-hunk gate (`crisp` default, `nl` judgment, or `shadow` agreement tracking). */
export function createTrivialFilter(setting?: PluggableSetting) {
  return pluggable<[Hunk], boolean>(
    { crisp: isTrivialHunkCrisp, nl: hunk => isTrivialHunkNl(hunk) },
    setting,
    { name: 'pr-shepherd.isTrivialHunk', default: 'crisp', serve: 'crisp' },
  );
}

/** Exact default verifier for doc-fix proposals: target file only, non-empty, no TODO placeholders, references findings. */
export async function defaultVerifyDocFix(
  proposal: FolderProposal<DocFixDraft>,
  findings: Finding[],
  targetPath: string,
): Promise<string[]> {
  const problems: string[] = [];
  for (const change of proposal.diff.changes) {
    if (change.path !== targetPath) problems.push(`Proposal modified unexpected path ${change.path}; only ${targetPath} may change.`);
  }
  if (!proposal.folder.isFile(targetPath)) {
    problems.push(`Target file ${targetPath} is missing from proposal.`);
    return problems;
  }
  const text = (await proposal.folder.readText(targetPath)).trim();
  if (!text) problems.push(`${targetPath} is empty.`);
  if (/\bTODO\b/i.test(text)) problems.push(`${targetPath} contains an unresolved TODO placeholder.`);
  for (const finding of findings) {
    if (!text.includes(finding.rule)) problems.push(`${targetPath} does not document rule "${finding.rule}".`);
  }
  return problems;
}

/** Directory reducer applying `proposeDocFix.nl` on a private Folder draft. */
async function applyDocFix(
  folder: FolderHandle,
  findings: Finding[],
  targetPath: string,
  problems: string[],
): Promise<DocFixDraft> {
  const file = folder.file(targetPath);
  const existingDoc = await file.exists() ? await file.readText() : '';
  const draft = await proposeDocFix(findings, targetPath, existingDoc, problems);
  await folder.file(targetPath).writeText(`${draft.content.trimEnd()}\n`);
  return { ...draft, path: targetPath };
}

/** Review a pull request's hunks against team rules, rank findings, gate merge readiness, and optionally verify a doc fix. */
export async function shepherdPullRequest(
  hunks: Hunk[],
  rules: Rule[],
  options: ShepherdOptions = {},
): Promise<ShepherdReport> {
  const concurrency = options.concurrency ?? 6;
  const isTrivial = createTrivialFilter(options.trivialMode);

  const trivialFlags = await semanticMap(hunks, hunk => isTrivial(hunk), { concurrency });
  const hunksToReview: { index: number; hunk: Hunk }[] = [];
  const hunkSummaries: HunkSummary[] = new Array(hunks.length);

  for (let i = 0; i < hunks.length; i++) {
    const hunk = hunks[i]!;
    if (trivialFlags[i]) {
      hunkSummaries[i] = { file: hunk.file, classification: { kind: 'trivial', risk: 'low' }, crispSkipped: true };
    } else {
      hunksToReview.push({ index: i, hunk });
    }
  }

  // Classify every non-trivial hunk concurrently via finite-record `readout: decision`.
  const classes: HunkClass[] = await semanticMap(hunksToReview, ({ hunk }) => classifyHunk(hunk), { concurrency });
  const activeHunks: { hunk: Hunk; classification: HunkClass }[] = [];
  for (let i = 0; i < hunksToReview.length; i++) {
    const entry = hunksToReview[i]!;
    const classification = classes[i]!;
    hunkSummaries[entry.index] = { file: entry.hunk.file, classification, crispSkipped: false };
    if (classification.kind !== 'trivial') activeHunks.push({ hunk: entry.hunk, classification });
  }

  // Bind checkHunk once per team rule with `withContext`, preserving its `ensures: quotesMatch` verifier.
  const boundChecks = rules.map(rule => ({
    rule,
    run: withContext(checkHunk, { rule: `${rule.name}: ${rule.guidance}` }),
  }));

  const pairs = activeHunks.flatMap(({ hunk, classification }) =>
    boundChecks
      .filter(({ rule }) => !rule.appliesTo || rule.appliesTo.includes(classification.kind))
      .map(check => ({ hunk, check }))
  );

  const reviews = await semanticMap(pairs, ({ hunk, check }) => check.run(hunk), { concurrency });
  const rawFindings: Finding[] = reviews.flatMap(review => review.findings);

  const findings = rawFindings.length > 1
    ? (await semanticRank(rawFindings, 'most critical engineering risk to fix before merging')).ranked
    : rawFindings;

  // Asymmetric safety gate: escalate when unsure (< floor) OR when P(block) >= blockRiskFloor.
  const decideMerge = escalate(mergeVerdict, {
    to: options.escalateTo ?? 'teacher',
    floor: options.floor ?? 0.85,
    risk: { block: options.blockRiskFloor ?? 0.25 },
  });
  const decision = await decideMerge(findings);

  let docFix: DocFixResult | undefined;
  if (options.docWorkspace && options.docTargetPath && findings.length > 0) {
    const targetPath = options.docTargetPath;
    const verify = options.verifyDocFix
      ? (proposal: FolderProposal<DocFixDraft>) => options.verifyDocFix!(proposal, findings, targetPath)
      : (proposal: FolderProposal<DocFixDraft>) => defaultVerifyDocFix(proposal, findings, targetPath);
    const proposed = await proposeAndVerify<DocFixDraft>(options.docWorkspace, applyDocFix, {
      args: [findings, targetPath],
      attempts: options.docFixAttempts ?? 2,
      verify,
    });
    docFix = {
      accepted: proposed.accepted,
      path: proposed.value.path,
      summary: proposed.value.summary,
      attempts: proposed.attempts,
      problems: proposed.problems,
    };
  }

  const crispHunks = hunkSummaries.filter(h => h.crispSkipped).length;
  return {
    verdict: decision.value,
    confidence: decision.confidence,
    escalated: decision.escalated,
    ...(decision.reason ? { escalationReason: decision.reason } : {}),
    findings,
    hunks: hunkSummaries,
    crispHunks,
    semanticHunks: hunks.length - crispHunks,
    ...(docFix ? { docFix } : {}),
  };
}
