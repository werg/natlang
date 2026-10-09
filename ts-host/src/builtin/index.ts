/**
 * The natlang programs that ship with the runtime: `.nl` files in this folder, embedded by scripts/generate-builtin.mjs
 * (sources.generated.ts). This module is a leaf (it imports nothing from the runtime), so prompt modules can take text
 * from it; the loader that turns the files into callables is runtime/builtin.ts.
 */
import { BUILTIN_SOURCES } from './sources.generated.js';

export { BUILTIN_SOURCES };

/** The names of the built-in programs (file names without `.nl`). */
export const builtinNames = (): string[] => Object.keys(BUILTIN_SOURCES);

/** The exact text of a built-in `.nl` file. */
export function builtinSource(name: string): string {
  const source = Object.hasOwn(BUILTIN_SOURCES, name) ? BUILTIN_SOURCES[name] : undefined;
  if (source === undefined) throw new RangeError(`no built-in natlang program named ${JSON.stringify(name)}; the built-ins are ${builtinNames().join(', ')}`);
  return source;
}

/** A built-in program's instructions as the model reads them: the text after the frontmatter, without its leading and trailing newlines. */
export function builtinBody(name: string): string {
  const match = /^---\r?\n[\s\S]*?\r?\n---\r?\n?([\s\S]*)$/.exec(builtinSource(name));
  if (!match) throw new Error(`built-in program ${name} has no frontmatter`);
  return match[1]!.replace(/^\n+|\n+$/g, '');
}

/**
 * The body of the builtin `view(value, instructions?)` (DECISIONS.md 2026-10-09, "one summarizer family"): faithful
 * compression without instructions, what their purpose needs with them. The prompt piece `view`: the system text of
 * view's write site, where its Neuralese instance is a template write of this body.
 */
export const VIEW_PROMPT = builtinBody('view');
