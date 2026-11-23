/**
 * Release notes from the merged changes. The exact rules for a draft are TypeScript (`notesProblems`); writing and
 * fixing the draft are judgments, and `semanticRepair` runs the fix until the rules hold. The summary is revised with
 * `fixedPoint` until a revision no longer changes what it says.
 */
import { nl } from '@natlang/node';
import { fixedPoint, semanticRepair } from '@natlang/node/std';
import type { Change, Notes, Summary } from './types.js';

const MAX_WORDS = 30;

/** The change id a line starts with, such as #412, or '' when it starts with none. */
const idOf = (line: string) => /^#\d+/.exec(line)?.[0] ?? '';

/** What is wrong with a draft, exactly: a change it leaves out, an id that is not in the release, a line too long. */
export function notesProblems(notes: Notes, changes: Change[]): string[] {
  const problems: string[] = [];
  for (const change of changes)
    if (!notes.some(line => idOf(line) === change.id)) problems.push(`No line starts with ${change.id} (${change.title}).`);
  for (const line of notes) {
    const id = idOf(line);
    if (!changes.some(change => change.id === id))
      problems.push(`The line "${line}" does not start with the id of a change in this release: start it with one of ${changes.map(change => change.id).join(', ')}.`);
    if (line.split(/\s+/).length > MAX_WORDS) problems.push(`The line for ${id} is longer than ${MAX_WORDS} words.`);
  }
  return problems;
}

export type Drafted = { notes: Notes; problems: string[] };

/** Draft the notes and repair the draft until the exact rules hold, or report what is still wrong after three fixes. */
export async function draftNotes(changes: Change[]): Promise<Drafted> {
  const first: Notes = await nl`Write one line of release notes per change in changes: the change's id, then what it means for a user of the project.`();
  const fix = nl<(notes: Notes, problems: string[]) => Notes>`Rewrite notes so that none of problems applies any more. Keep every line that no problem names.`;
  const repaired = await semanticRepair(first, { check: notes => notesProblems(notes, changes), fix, attempts: 3 });
  return { notes: repaired.value, problems: repaired.problems };
}

/** The summary paragraph, revised until a revision only rewords it. */
export async function summarize(notes: Notes): Promise<Summary> {
  const first: Summary = await nl`Write the paragraph that tells a user of the project what changed, from notes.`();
  const revise = nl<(summary: Summary) => Summary>`Revise summary so that it states every change in notes that a user notices, and nothing that notes do not say.`;
  const { value } = await fixedPoint(revise, first);
  return value;
}
