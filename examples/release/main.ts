/**
 * `natlang run examples/release`: draft the release notes for changes.json, summarize them, merge the two
 * maintainers' edits of release.json, then write the release section and the announcement into the project folder
 * under the program's state directory through verified transactions. Prints what happened at each stage.
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { openFolder, saveFolder, untrusted, type TargetContext } from '@natlang/node';
import { draftNotes, summarize } from './notes.js';
import { mergeRelease } from './merge.js';
import { announce, publish } from './publish.js';
import type { Change, Release } from './types.js';

export async function main(context: TargetContext): Promise<number> {
  const root = context.package?.root ?? context.workspace;
  const read = (path: string) => JSON.parse(readFileSync(join(root, path), 'utf8'));
  // The change descriptions are their authors' text: outside data from here on.
  const changes = (read('changes.json') as Change[]).map(change => ({ ...change, description: untrusted(change.description, `change ${change.id}`) }));
  const { base, ours, theirs } = read('release.json') as Record<'base' | 'ours' | 'theirs', Release>;
  const project = join(context.stateDirectory, 'project');
  mkdirSync(project, { recursive: true });
  if (!existsSync(join(project, 'README.md'))) writeFileSync(join(project, 'README.md'), '# Sync\n');
  const folder = openFolder(project, 'overlay');
  const report = await context.runtime.run(async () => {
    const drafted = await draftNotes(changes);
    const [summary, release] = await Promise.all([summarize(drafted.notes), mergeRelease(base, ours, theirs)]);
    const version = release.value.version;
    const section = await publish(folder, version, drafted.notes);
    const announcement = await announce(folder, drafted.notes);
    return { notes: drafted, summary, release, section: { accepted: section.accepted, problems: section.problems, attempts: section.attempts },
      announcement: { chosen: announcement.chosen, text: announcement.value } };
  });
  saveFolder(project, folder);
  context.io.output.write(JSON.stringify(report, null, 2) + '\n');
  return 0;
}
