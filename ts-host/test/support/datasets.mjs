/**
 * Tests drawn from external datasets run only where the dataset cache has them (NATLANG_DATASETS, default
 * vendor/datasets). `missingDatasets('folio')` is a skip reason naming the acquire command, or false.
 */
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const CACHE = process.env.NATLANG_DATASETS ?? fileURLToPath(new URL('../../../vendor/datasets', import.meta.url));

export function missingDatasets(...sources) {
  const missing = sources.filter(source => !existsSync(join(CACHE, source)));
  return missing.length ? `not in the dataset cache: ${missing.join(', ')} ` +
    `(node scripts/inline-curriculum/acquire.mjs --source ${missing[0]})` : false;
}

/** An error a dataset-backed curriculum family throws when its data is not present on this machine. */
export const missingDataError = error => /not in the dataset cache|is not in vendor\/datasets|missing; sync|not present; sync/.test(String(error?.message));
