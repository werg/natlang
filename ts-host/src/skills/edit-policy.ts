/** Host-side constraints for genuine metadata-only skill ablations. */
import { isDeepStrictEqual } from 'node:util';
import { splitFrontmatter } from './skill.js';

export function checkSkillMetadataOnlyEdit(baseline: Readonly<Record<string,string>>, candidate: Readonly<Record<string,string>>,
  paths: readonly string[]): string[] {
  const diagnostics: string[] = [];
  for (const path of paths) {
    try {
      const before = splitFrontmatter(baseline[path] ?? ''), after = splitFrontmatter(candidate[path] ?? '');
      if (!before || !after || !before.data || !after.data || typeof before.data !== 'object' || typeof after.data !== 'object')
        throw Error('a skill document was added, removed or malformed');
      const frozenMeta = (value: unknown) => Object.fromEntries(Object.entries(value as Record<string,unknown>)
        .filter(([key]) => key !== 'description' && key !== 'summary'));
      if (before.body !== after.body) diagnostics.push(`${path}: metadata-only tuning cannot change skill instructions`);
      if (!isDeepStrictEqual(frozenMeta(before.data), frozenMeta(after.data)))
        diagnostics.push(`${path}: metadata-only tuning can change only description and summary`);
    } catch (error) { diagnostics.push(`${path}: ${String(error)}`); }
  }
  return diagnostics;
}
