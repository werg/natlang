/**
 * Publishing writes files through transactions. `proposeAndVerify` writes the release section on a private copy of the
 * project folder, checks it with an exact verifier the reducer cannot edit, and accepts it only when it passes.
 * `branchAndSelect` drafts the announcement in two styles, each on its own copy, and keeps the one that serves the
 * stated goal better.
 */
import { nl, type Folder, type FolderHandle, type FolderProposal } from '@natlang/node';
import { branchAndSelect, proposeAndVerify } from '@natlang/node/std';
import type { Notes } from './types.js';

/** The reducer: write releases/VERSION.md from the notes. On a retry it reads its earlier section and the problems. */
export async function writeSection(folder: FolderHandle, version: string, notes: Notes, problems: string[]): Promise<string> {
  const file = folder.file(`releases/${version}.md`);
  const previous = await file.exists() ? await file.readText() : '';
  const section: string = await nl`Write the release section for version from notes: a first line "## " followed by version, then one line per note that starts with "- ". If previous is not empty, it is your earlier section and problems says what is wrong with it: fix exactly that.`();
  await file.writeText(`${section.trim()}\n`);
  return section;
}

/** The verifier, exact: the heading, one item per note, and no other file changed. */
export async function sectionProblems(proposal: FolderProposal<string>, version: string, notes: Notes): Promise<string[]> {
  const path = `releases/${version}.md`;
  const problems = proposal.diff.changes.filter(change => change.path !== path).map(change => `The change also touches ${change.path}.`);
  const [heading, ...rest] = (await proposal.folder.readText(path)).trim().split('\n');
  if (heading !== `## ${version}`) problems.push(`The first line is "${heading}", not "## ${version}".`);
  const items = rest.filter(line => line.trim());
  if (items.some(line => !line.startsWith('- '))) problems.push('A line after the heading does not start with "- ".');
  if (items.length !== notes.length) problems.push(`The section has ${items.length} items for ${notes.length} notes.`);
  return problems;
}

export async function publish(folder: Folder, version: string, notes: Notes) {
  return proposeAndVerify<string>(folder, writeSection, { args: [version, notes], attempts: 2,
    verify: proposal => sectionProblems(proposal, version, notes) });
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
