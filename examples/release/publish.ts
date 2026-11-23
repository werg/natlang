/**
 * Publishing writes files through transactions. `proposeAndVerify` updates CHANGELOG.md and drafts a migration guide
 * for breaking changes in MIGRATION.md on a private copy of the folder, checks the proposal with an exact verifier the
 * reducer cannot edit (prior releases untouched, every breaking change documented, no stray files), and accepts only
 * when it passes. `branchAndSelect` drafts the announcement in two styles, each on its own copy, and keeps the one
 * that serves the stated goal better via debiased pairwise ranking.
 */
import { nl, type Folder, type FolderHandle, type FolderProposal } from '@natlang/node';
import { branchAndSelect, proposeAndVerify } from '@natlang/node/std';
import type { Change, Notes } from './types.js';

/** Deterministic Markdown formatting stays in TypeScript; only the breaking-change migration prose calls the model. */
export async function writeReleaseDocs(
  folder: FolderHandle,
  version: string,
  notes: Notes,
  breaking: Change[],
  problems: string[],
): Promise<string> {
  const changelog = folder.file('CHANGELOG.md');
  const existing = await changelog.exists() ? await changelog.readText() : '';
  const section = `## ${version}\n${notes.map(note => `- ${note}`).join('\n')}\n`;
  // Exact prepend in TypeScript keeps every prior release byte-for-byte intact.
  const nextChangelog = existing.includes(`## ${version}\n`) ? existing : `${section}\n${existing}`.trimEnd() + '\n';
  await changelog.writeText(nextChangelog);

  const migration = folder.file('MIGRATION.md');
  const previous = await migration.exists() ? await migration.readText() : '';
  const guide: string = await nl`Write a concise upgrade guide for version covering every breaking change in breaking, using notes for context. Name the exact removed or replaced APIs and what callers should switch to. If previous is not empty, it is your earlier draft and problems lists what the verifier found wrong: fix every problem.`();
  await migration.writeText(`${guide.trim()}\n`);
  return guide;
}

/** Exact verifier the reducer cannot edit: prior changelog entries preserved, breaking changes covered, no stray files. */
export async function releaseDocProblems(
  proposal: FolderProposal<string>,
  version: string,
  priorChangelog: string,
  breaking: Change[],
): Promise<string[]> {
  const allowed = new Set(['CHANGELOG.md', 'MIGRATION.md']);
  const problems = proposal.diff.changes
    .filter(change => !allowed.has(change.path))
    .map(change => `The change also touches ${change.path}.`);
  const changelog = await proposal.folder.readText('CHANGELOG.md');
  if (!changelog.includes(`## ${version}\n`)) problems.push(`CHANGELOG.md is missing the "## ${version}" heading.`);
  if (priorChangelog.trim() && !changelog.includes(priorChangelog.trim()))
    problems.push('CHANGELOG.md altered or dropped an earlier release section.');
  const guide = (await proposal.folder.readText('MIGRATION.md')).trim();
  if (!guide) problems.push('MIGRATION.md is empty.');
  if (/\bTODO\b/i.test(guide)) problems.push('MIGRATION.md still contains a TODO placeholder.');
  for (const change of breaking) {
    if (!guide.includes(change.id)) problems.push(`MIGRATION.md does not reference breaking change ${change.id} (${change.title}).`);
  }
  return problems;
}

export async function publish(folder: Folder, version: string, notes: Notes, breaking: Change[] = []) {
  const priorChangelog = await folder.file('CHANGELOG.md').exists() ? await folder.file('CHANGELOG.md').readText() : '';
  return proposeAndVerify<string>(folder, writeReleaseDocs, {
    args: [version, notes, breaking],
    attempts: 2,
    verify: proposal => releaseDocProblems(proposal, version, priorChangelog, breaking),
  });
}

/** An announcement writer for one style. */
const inStyle = (style: string) => async (folder: FolderHandle, notes: Notes): Promise<string> => {
  const text: string = await nl`Write a short release announcement from notes, in the style that style describes.`();
  await folder.file('ANNOUNCEMENT.md').writeText(`${text.trim()}\n`);
  return text;
};

export async function announce(folder: Folder, notes: Notes) {
  return branchAndSelect<string>(folder, [
    { reducer: inStyle('plain sentences, most important change first'), args: [notes] },
    { reducer: inStyle('a bulleted list, one bullet per note'), args: [notes] },
  ], {
    verify: async proposal => (await proposal.folder.readText('ANNOUNCEMENT.md')).trim() ? [] : ['The announcement is empty.'],
    goal: 'a user who is about to upgrade learns first what will break and what to do about it',
  });
}
