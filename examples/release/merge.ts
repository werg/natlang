/**
 * Two maintainers edited the release record at the same time. `semanticMerge` takes each one's edit where the other
 * left a field alone, judges the fields both changed by what each edit is for, combines the compatible ones and lists
 * the rest. The version stays exact: a combined version must still be a version number.
 */
import { semanticMerge, type Merge } from '@natlang/node/std';
import type { Release } from './types.js';

export function mergeRelease(base: Release, ours: Release, theirs: Release): Promise<Merge<Release>> {
  return semanticMerge(base, ours, theirs, {
    check: (value, path) => path[0] === 'version' && !/^\d+\.\d+\.\d+$/.test(String(value)) ? [`${String(value)} is not a version number such as 2.4.0.`] : [],
  });
}
