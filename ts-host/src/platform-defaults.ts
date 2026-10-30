/**
 * The platform-bound entry points of a platform entry (`@natlang/node`, `@natlang/browser`): the kernel functions that
 * need a platform, with the entry's own as the default. A caller passes `platform` to run on another.
 */
import { createNatlangRuntime, type NatlangRuntime, type NatlangRuntimeOptions } from './runtime/runtime.js';
import type { Platform } from './runtime/platform.js';
import { compileVirtualProject, loadVirtualCallables, loadVirtualNatlang, type CompiledProject, type VirtualProject,
  type VirtualProjectOptions } from './runtime/virtual-project.js';
import { projectEntry } from './app/playground.js';
import type { NatlangCallable } from './runtime/callable.js';

export function platformDefaults(defaultPlatform: () => Platform) {
  return {
    /** A runtime on the entry's platform unless `platform` says otherwise. */
    createNatlangRuntime: (options: NatlangRuntimeOptions = {}): NatlangRuntime => createNatlangRuntime({ platform: defaultPlatform(), ...options }),
    /** Compile an in-memory project whose modules run on the entry's platform unless `platform` says otherwise. */
    compileVirtualProject: (project: VirtualProject, runtime: Record<string, unknown>,
        options: Omit<VirtualProjectOptions, 'platform'> & { platform?: Platform } = {}): CompiledProject =>
      compileVirtualProject(project, runtime, { platform: defaultPlatform(), ...options }),
    /** The callable entry of a playground project file; TypeScript entries run on the entry's platform unless `platform` says otherwise. */
    projectEntry: (files: Record<string, string>, path: string, runtimeNamespace?: Record<string, unknown>,
        platform: Platform = defaultPlatform()) => projectEntry(files, path, runtimeNamespace, platform),
    loadVirtualNatlang: (files: Record<string, string>, path: string, root = '/project', platform: Platform = defaultPlatform()): NatlangCallable =>
      loadVirtualNatlang(files, path, root, platform),
    loadVirtualCallables: (files: Record<string, string>, dir = '', root = '/project', platform: Platform = defaultPlatform()): Record<string, unknown> =>
      loadVirtualCallables(files, dir, root, platform),
  };
}
