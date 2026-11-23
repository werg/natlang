/** `natlang run examples/outline`: digest the bundled document.json, section by section. */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { untrusted, type TargetContext } from '@natlang/node';
import { outline } from './outline.js';
import type { Section } from './types.js';

/** The document's text is outside data: every section's text is marked where it enters. */
function marked(section: Section): Section {
  return { ...section, text: untrusted(section.text, `section ${section.title}`), sections: section.sections.map(marked) };
}

export async function main(context: TargetContext): Promise<number> {
  const root = context.package?.root ?? context.workspace;
  const document = JSON.parse(readFileSync(join(root, 'document.json'), 'utf8')) as Section;
  const { contents } = await context.runtime.run(() => outline(marked(document)));
  context.io.output.write(contents.join('\n') + '\n');
  return 0;
}
