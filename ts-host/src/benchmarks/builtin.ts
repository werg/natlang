/**
 * The benchmark packages that ship with the host. Importing this module registers them (idempotently); core
 * modules import it once so that specs naming a benchmark comparator or contract resolve identically everywhere.
 * To add a benchmark: create `benchmarks/<name>/index.ts` exporting a `register<Name>()` and list it here.
 */
import { registerMusique } from './musique/index.js';
import { registerTatqa } from './tatqa/index.js';

export const BUILTIN_BENCHMARKS = ['tatqa', 'musique'] as const;
registerTatqa();
registerMusique();
