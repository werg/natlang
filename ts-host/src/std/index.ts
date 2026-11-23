/**
 * `@natlang/std`, as `@natlang/node/std` and `@natlang/browser/std` (library.ts). Loading it registers the library for
 * the programs the runtime compiles in memory (virtual projects, callable-folder modules), so they import the same
 * module instance as the host.
 */
import { STD_SPECIFIERS } from '../compiler/intrinsics.js';
import { registerShippedModule } from '../runtime/modules.js';
import * as library from './library.js';

export * from './library.js';
registerShippedModule(STD_SPECIFIERS, () => ({ __esModule: true, ...library }));
