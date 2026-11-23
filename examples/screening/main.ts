/** `natlang run examples/screening`: screen the bundled applications.json for the role it names. */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { untrusted, type TargetContext } from '@natlang/node';
import { screen } from './screen.js';
import type { Application } from './types.js';

export async function main(context: TargetContext): Promise<number> {
  const root = context.package?.root ?? context.workspace;
  const { role, requirements, applications } = JSON.parse(readFileSync(join(root, 'applications.json'), 'utf8')) as
    { role: string; requirements: string[]; applications: Application[] };
  // The applicants' text enters here: mark it as outside data, labelled with where it came from.
  const marked = applications.map(application => ({ ...application, text: untrusted(application.text, `application ${application.id}`) }));
  const screening = await context.runtime.run(() => screen(marked, role, requirements));
  context.io.output.write(JSON.stringify(screening, null, 2) + '\n');
  return 0;
}
