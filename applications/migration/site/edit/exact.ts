/**
 * Whether a site's patches are exact, a pluggable hot path of every patch. The setting
 * `repository.implementation('exact')` selects the counts below (crisp) or `exact/judge.nl` (natural-language), which
 * also judges that the patches change only what the site's usage calls for.
 */
import { pluggable } from '@natlang/node';
import { repository } from 'natlang:services';
import judge from './exact/judge.nl';
import type { Classified, Exactness, Patch } from '../../types.js';

/** Every patch is in the site's file, changes something, and has an old text that occurs once at the revision. */
export function crisp(patches: Patch[], classified: Classified, revision: string): Exactness {
  if (!patches.length) return { exact: false, problem: `no patch for site ${classified.site.id} (${classified.site.path}:${classified.site.from}); write the edit its usage calls for` };
  for (const patch of patches) {
    if (patch.path !== classified.site.path) return { exact: false, problem: `patch path ${patch.path} is not the site's file ${classified.site.path}` };
    if (!patch.old || patch.old === patch.new) return { exact: false, problem: `patch in ${patch.path} must have a non-empty old text and a different new text` };
    const found = repository.count(patch.path, patch.old, revision);
    if (found !== 1) return { exact: false, problem: `old text occurs ${found} times in ${patch.path}: ${JSON.stringify(patch.old.slice(0, 80))}; extend it with neighboring text until it occurs once` };
  }
  return { exact: true, problem: '' };
}

export default async function exact(patches: Patch[], classified: Classified, revision: string): Promise<Exactness> {
  return pluggable({ crisp: () => crisp(patches, classified, revision), nl: () => judge(patches, classified, revision) },
    await repository.implementation('exact'), { name: 'migration.exact', same: (exact, judged) => exact.exact === judged.exact })();
}
