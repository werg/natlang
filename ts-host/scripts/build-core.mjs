#!/usr/bin/env node
/**
 * The workerd build of natlang's platform-neutral core: dist/core/natlang-core.js (one ESM bundle of src/core/index.ts),
 * dist/core/natlang-core.d.ts and the declarations it re-exports (dist/core/types/). The bundle runs in workerd with
 * `nodejs_compat`; the npm packages it imports stay external for the consumer to install. See scripts/vendor-build.mjs.
 *
 *   node scripts/build-core.mjs
 */
import { join } from 'node:path';
import { buildTarget, root } from './vendor-build.mjs';

try {
  const { files } = await buildTarget('core', join(root, 'dist/core'));
  console.log(`core built into dist/core: ${files.length} files`);
} catch (error) {
  console.error(error.errors ? error.errors.map(item => item.text).join('\n') : error);
  process.exit(1);
}
