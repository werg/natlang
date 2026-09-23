/**
 * Compile and execute a project held in memory (browser playground, Studio workers, tests).
 * The project compiles to CommonJS and runs through a small module loader in this realm, so it
 * shares the calling runtime instance.
 */
import { compileProject, type BuildResult, type ProjectFiles } from '../compiler/project.js';
import { loadCallableFolder, loadNamedFunction, type SourceFiles } from './loader.js';
import { callableTree, namedCallable, type NatlangCallable } from './callable.js';

export type VirtualProject = { files: Record<string, string>; root?: string };

/** In-memory project files (absolute POSIX paths). */
export function virtualProjectFiles(files: Record<string, string>, root = '/project'): ProjectFiles & { outputs: Map<string, string> } {
  const store = new Map(Object.entries(files).map(([path, text]) => [path.startsWith('/') ? path : `${root}/${path}`, text]));
  const outputs = new Map<string, string>();
  const directories = () => {
    const dirs = new Set<string>(['/']);
    for (const path of [...store.keys(), ...outputs.keys()]) {
      let at = path.lastIndexOf('/');
      while (at > 0) { dirs.add(path.slice(0, at)); at = path.lastIndexOf('/', at - 1); }
    }
    return dirs;
  };
  return { outputs,
    isFile: path => store.has(path) || outputs.has(path),
    isDirectory: path => directories().has(path.replace(/\/$/, '') || '/'),
    list: dir => { const prefix = dir.replace(/\/$/, '') + '/';
      return [...new Set([...store.keys(), ...outputs.keys()].filter(path => path.startsWith(prefix))
        .map(path => path.slice(prefix.length).split('/')[0]!))]; },
    read: path => { const text = store.get(path) ?? outputs.get(path); if (text === undefined) throw new Error(`no such file ${path}`); return text; },
    write: (path, text) => { outputs.set(path, text); } };
}

/** Loader access to in-memory files; display paths are relative to `root`. */
export function virtualSourceFiles(files: Record<string, string>, root = '/project'): SourceFiles {
  const virtual = virtualProjectFiles(files, root);
  return { join: (...parts) => parts.join('/').replace(/\/+/g, '/'),
    dirname: path => path.slice(0, path.lastIndexOf('/')) || '/',
    basename: (path, extension) => { const base = path.slice(path.lastIndexOf('/') + 1); return extension && base.endsWith(extension) ? base.slice(0, -extension.length) : base; },
    extname: path => { const base = path.slice(path.lastIndexOf('/') + 1); const at = base.lastIndexOf('.'); return at > 0 ? base.slice(at) : ''; },
    isFile: path => virtual.isFile(path), isDirectory: path => virtual.isDirectory(path),
    read: path => virtual.read(path), list: path => virtual.list(path),
    relative: path => path.startsWith(`${root}/`) ? path.slice(root.length + 1) : path };
}

/** Load a named `.nl` function, with its companion folder, from in-memory files. */
export function loadVirtualNatlang(files: Record<string, string>, path: string, root = '/project'): NatlangCallable {
  const record = loadNamedFunction(path.startsWith('/') ? path : `${root}/${path}`, virtualSourceFiles(files, root));
  return namedCallable(record.name, record);
}

/** Load an in-memory directory as a callable folder: its `.nl` and `.ts` items as a record of callables. */
export function loadVirtualCallables(files: Record<string, string>, dir = '', root = '/project'): Record<string, unknown> {
  return callableTree(loadCallableFolder(dir ? `${root}/${dir}` : root, virtualSourceFiles(files, root)));
}

export type CompiledProject = BuildResult & { require(path: string): Record<string, unknown> };

/**
 * Compile a virtual project and return a loader for its modules. `runtime` is the namespace that
 * `@natlang/browser` (and any `runtimeSpecifiers`) resolve to inside the project.
 */
export function compileVirtualProject(project: VirtualProject, runtime: Record<string, unknown>,
  options: { runtimeSpecifiers?: string[]; target?: 'node' | 'browser'; modules?: Record<string, unknown> } = {}): CompiledProject {
  const root = project.root ?? '/project';
  const files = virtualProjectFiles(project.files, root);
  const specifiers = options.runtimeSpecifiers ?? ['@natlang/browser', '@natlang/node'];
  const result = compileProject({ project: root, files, outDir: `${root}/.natlang/build`, module: 'commonjs',
    target: options.target ?? 'browser', runtimeSpecifier: specifiers[0]!, write: true, surfaceSpecifiers: specifiers });
  const cache = new Map<string, { exports: Record<string, unknown> }>();
  const load = (path: string): Record<string, unknown> => {
    const found = cache.get(path);
    if (found) return found.exports;
    const text = result.outputs[path];
    if (text === undefined) throw new Error(`module ${path} was not emitted by the project build`);
    const module = { exports: {} as Record<string, unknown> };
    cache.set(path, module);
    const require = (specifier: string): unknown => {
      if (specifiers.includes(specifier)) return runtime;
      if (options.modules && Object.hasOwn(options.modules, specifier)) return options.modules[specifier];
      if (!specifier.startsWith('.')) throw new Error(`module ${specifier} is not available in this virtual project`);
      const base = path.slice(0, path.lastIndexOf('/'));
      const joined = `${base}/${specifier}`.split('/').reduce<string[]>((parts, part) =>
        part === '..' ? parts.slice(0, -1) : part === '.' || !part ? parts : [...parts, part], []).join('/');
      const candidate = ['', '.js', '/index.js'].map(suffix => `/${joined.replace(/\.ts$/, '.js')}${suffix}`).find(item => item in result.outputs);
      if (!candidate) throw new Error(`cannot resolve ${specifier} from ${path}`);
      return load(candidate);
    };
    new Function('exports', 'require', 'module', text)(module.exports, require, module);
    return module.exports;
  };
  return { ...result, require: (path: string) => {
    const absolute = path.startsWith('/') ? path : `${root}/${path}`;
    const emitted = `${root}/.natlang/build/${absolute.slice(root.length + 1)}`.replace(/\.m?ts$/, '.js');
    return load(emitted);
  } };
}
