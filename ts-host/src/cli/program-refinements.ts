/**
 * Crisp checkers of a program's refinement predicates (plans/REFINEMENT_TYPES.md section 3). A program that wants some
 * `Is<T, "predicate">` decided exactly writes an ordinary TypeScript module `refinements.ts` beside its entry module (or
 * in a directory above it, within the package) that exports
 *
 *     export const refinements = { "one line of at most 60 characters": (value) => typeof value === "string" && value.length <= 60 };
 *
 * The build compiles it with the rest of the program; the launcher imports the emitted module and hands the table to the
 * runtime (`refinements.crisp`). It is host code like the entry module, not callable-folder code, so it needs no
 * finite-iteration policy. A checker returns a boolean to decide or `undefined` to defer to the judge.
 */
import { existsSync } from 'node:fs';
import { dirname, join, resolve, sep } from 'node:path';
import { pathToFileURL } from 'node:url';
import { normalizePredicate } from '../native/refinement-settings.js';

export type CrispCheckers = Record<string, (value: unknown) => boolean | undefined>;

/** The emitted `refinements.js` nearest the compiled entry module, not leaving `stop`. */
export function findRefinementsModule(compiledEntry: string, stop: string): string | undefined {
  const top = resolve(stop);
  for (let dir = dirname(resolve(compiledEntry)); dir === top || dir.startsWith(top + sep); dir = dirname(dir)) {
    const candidate = join(dir, 'refinements.js');
    if (existsSync(candidate)) return candidate;
    if (dir === top || dirname(dir) === dir) break;
  }
  return undefined;
}

/** Import the program's crisp checkers, keyed by normalized predicate. Nothing when the program has none. */
export async function loadProgramRefinements(compiledEntry: string, stop: string): Promise<CrispCheckers | undefined> {
  const path = findRefinementsModule(compiledEntry, stop);
  if (!path) return undefined;
  const module = await import(pathToFileURL(path).href) as { refinements?: unknown };
  const table = module.refinements;
  if (!table || typeof table !== 'object' || Array.isArray(table))
    throw new TypeError(`${path} must export \`refinements\`: an object from predicate text to (value) => boolean | undefined`);
  const out: CrispCheckers = Object.create(null);
  for (const [predicate, checker] of Object.entries(table)) {
    if (typeof checker !== 'function') throw new TypeError(`${path}: refinements[${JSON.stringify(predicate)}] must be a function`);
    out[normalizePredicate(predicate)] = checker as CrispCheckers[string];
  }
  return out;
}
