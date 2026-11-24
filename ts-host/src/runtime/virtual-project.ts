import {invokeNativeEntry} from './native-entry.js';
/**
 * Compile and execute a project held in memory (browser playground, Studio workers, tests).
 * The project compiles to CommonJS and runs through a small module loader in this realm, so it
 * shares the calling runtime instance.
 */
import { compileProject, type BuildResult, type ProjectFiles } from '../compiler/project.js';
import { NEUTRAL_RUNTIME_SPECIFIER } from '../compiler/runtime-specifier.js';
import { loadCallableFolder, loadNamedFunction, parseModule, type SourceFiles } from './loader.js';
import { joinSourcePath } from './source-path.js';
import { callableTree, namedCallable, type NatlangCallable } from './callable.js';
import { builtinModule, hasBuiltinModule, moduleTarget, platformModuleRealm, servicesModule } from './modules.js';
import type { Platform } from './platform.js';
import type { ProgramDescriptor } from '../adaptation/types.js';

export type VirtualProject = { files: Record<string, string>; root?: string };

/** In-memory project files (absolute POSIX paths). */
export function virtualProjectFiles(files: Record<string, string>, root = '/project'): ProjectFiles & { outputs: Map<string, string> } {
  const store = new Map(Object.entries(files).map(([path, text]) => [path.startsWith('/') ? path : `${root}/${path}`, text]));
  const outputs = new Map<string, string>();
  const sourceChildren = new Map<string, Set<string>>([['/', new Set()]]);
  for (const path of store.keys()) {
    let child = path;
    for (let slash = child.lastIndexOf('/'); slash >= 0; slash = child.lastIndexOf('/')) {
      const parent = child.slice(0, slash) || '/';
      let entries = sourceChildren.get(parent);
      if (!entries) {
        entries = new Set();
        sourceChildren.set(parent, entries);
      }
      entries.add(child.slice(slash + 1));
      if (slash === 0) break;
      child = parent;
    }
  }
  const directory = (path: string) => path.replace(/\/$/, '') || '/';
  const prefixOf = (path: string) => path === '/' ? '/' : path + '/';
  return { outputs,
    isFile: path => store.has(path) || outputs.has(path),
    isDirectory: path => {
      const dir = directory(path);
      if (sourceChildren.has(dir)) return true;
      const prefix = prefixOf(dir);
      for (const output of outputs.keys()) if (output.startsWith(prefix)) return true;
      return false;
    },
    list: path => {
      const dir = directory(path);
      if (!outputs.size) return [...(sourceChildren.get(dir) ?? [])];
      const entries = new Set(sourceChildren.get(dir));
      const prefix = prefixOf(dir);
      for (const output of outputs.keys())
        if (output.startsWith(prefix)) entries.add(output.slice(prefix.length).split('/')[0]!);
      return [...entries];
    },
    read: path => { const text = store.get(path) ?? outputs.get(path); if (text === undefined) throw new Error(`no such file ${path}`); return text; },
    write: (path, text) => { outputs.set(path, text); } };
}

/** Loader access to in-memory files; display paths are relative to `root`. */
export function virtualSourceFiles(files: Record<string, string>, root = '/project'): SourceFiles {
  const virtual = virtualProjectFiles(files, root);
  return { join: joinSourcePath,
    dirname: path => path.slice(0, path.lastIndexOf('/')) || '/',
    basename: (path, extension) => { const base = path.slice(path.lastIndexOf('/') + 1); return extension && base.endsWith(extension) ? base.slice(0, -extension.length) : base; },
    extname: path => { const base = path.slice(path.lastIndexOf('/') + 1); const at = base.lastIndexOf('.'); return at > 0 ? base.slice(at) : ''; },
    isFile: path => virtual.isFile(path), isDirectory: path => virtual.isDirectory(path),
    read: path => virtual.read(path), list: path => virtual.list(path),
    relative: path => path.startsWith(`${root}/`) ? path.slice(root.length + 1) : path, root };
}

/** Load a named `.nl` function, with its companion folder, from in-memory files; its TypeScript modules run on `platform`. */
export function loadVirtualNatlang(files: Record<string, string>, path: string, root = '/project', platform?: Platform): NatlangCallable {
  const record = loadNamedFunction(path.startsWith('/') ? path : `${root}/${path}`, virtualSourceFiles(files, root));
  return namedCallable(record.name, record, undefined, platform);
}

/** Load an in-memory directory as a callable folder: its `.nl` and `.ts` items as a record of callables. */
export function loadVirtualCallables(files: Record<string, string>, dir = '', root = '/project', platform?: Platform): Record<string, unknown> {
  return callableTree(loadCallableFolder(dir ? `${root}/${dir}` : root, virtualSourceFiles(files, root)), undefined, platform);
}

export type CompiledProject = BuildResult & { require(path: string): Record<string, unknown> };

export type VirtualProjectOptions = { inlineRecords?: boolean; platform: Platform; constrained?: boolean; programId?: string; guidance?: string;
  importedGuidancePrograms?: readonly string[]; services?: ProgramDescriptor['services']; runtimeSpecifiers?: string[];
  target?: 'node' | 'browser'; modules?: Record<string, unknown>;
  /** Validate source and produce declarations without emitting JavaScript. */
  emit?: boolean;
  /** Collect the program descriptor for adaptation. Defaults to true. */
  adaptation?: boolean };

/**
 * Compile a virtual project and return a loader for its modules. `runtime` is the namespace that
 * `@natlang/browser` (and any `runtimeSpecifiers`) resolve to inside the project. The modules run in `platform`'s
 * module realm (the Function constructor, a vm, a host's confined realm) and resolve its platform modules.
 *
 * Compiled code refers to the runtime only through `require(specifier)` for the specifiers in `runtimeSpecifiers`
 * (default `@natlang/browser`, `@natlang/node`) and the neutral `NEUTRAL_RUNTIME_SPECIFIER`; the loader answers those
 * with `runtime`, so a host binds the project to its one runtime instance by passing it here. `target` says whether
 * awaits restore the natlang task (`browser`); by default the platform's context store decides.
 */
/** A serializable frontend product. It contains no runtime, service, realm or module instance. */
export type VirtualProjectProduct = {
  schema: 'natlang.virtual-project-product/1';
  project: VirtualProject;
  runtimeSpecifiers: string[];
  build: BuildResult;
};
export type VirtualProjectProductOptions = Omit<VirtualProjectOptions, 'platform' | 'modules' | 'target'> & {
  /** Explicit: a compiler worker has no runtime platform from which to infer this. */
  target: 'node' | 'browser';
};

/** Pure frontend compilation, suitable for an isolated worker and an exact-input shared cache. */
export function compileVirtualProjectProduct(project: VirtualProject, options: VirtualProjectProductOptions): VirtualProjectProduct {
  const snapshot = structuredClone(project);
  const root = snapshot.root ?? '/project';
  const files = virtualProjectFiles(snapshot.files, root);
  const specifiers = [...(options.runtimeSpecifiers ?? ['@natlang/browser', '@natlang/node'])];
  const build = compileProject({ project: root, constrained: options.constrained, programId: options.programId, files, outDir: `${root}/.natlang/build`, module: 'commonjs',
    guidance: options.guidance, importedGuidancePrograms: options.importedGuidancePrograms, services: options.services,
    inlineRecords:options.inlineRecords,target: options.target, runtimeSpecifier: specifiers[0]!, write: true, emit: options.emit, adaptation: options.adaptation, surfaceSpecifiers: specifiers });
  return { schema: 'natlang.virtual-project-product/1', project: snapshot, runtimeSpecifiers: specifiers, build };
}

/** Bind a reusable frontend product to this owner's runtime and fresh module/realm state. */
export function instantiateVirtualProjectProduct(product: VirtualProjectProduct, runtime: Record<string, unknown>, options: Pick<VirtualProjectOptions, 'platform' | 'modules'>): CompiledProject {
  if (product.schema !== 'natlang.virtual-project-product/1') throw new Error('unsupported virtual project product');
  // Public build metadata stays caller-owned; executable closures only own the
  // source and emitted modules they can use after that metadata is released.
  const { project, build: result, runtimeSpecifiers: specifiers } = structuredClone(product);
  return { ...result, require: createVirtualProjectRequire(project, result.outputs, specifiers, runtime, options) };
}

function createVirtualProjectRequire(
  project: VirtualProject, outputs: Record<string, string>, specifiers: readonly string[],
  runtime: Record<string, unknown>, options: Pick<VirtualProjectOptions, 'platform' | 'modules'>
): CompiledProject['require'] {
  const { platform } = options;
  const root = project.root ?? '/project';
  const cache = new Map<string, { exports: Record<string, unknown> }>();
  const load = (path: string): Record<string, unknown> => {
    const found = cache.get(path);
    if (found) return found.exports;
    const text = outputs[path];
    if (text === undefined) throw new Error(`module ${path} was not emitted by the project build`);
    const module = { exports: {} as Record<string, unknown> };
    cache.set(path, module);
    const require = (specifier: string): unknown => {
      if (specifier === 'natlang:services') return servicesModule;
      if (specifiers.includes(specifier) || specifier === NEUTRAL_RUNTIME_SPECIFIER) return runtime;
      if (options.modules && Object.hasOwn(options.modules, specifier)) return options.modules[specifier];
      if (hasBuiltinModule(platform, specifier)) return builtinModule(platform, specifier);
      if (!specifier.startsWith('.')) throw new Error(`module ${specifier} is not available in this virtual project`);
      const base = path.slice(0, path.lastIndexOf('/'));
      const joined = joinSourcePath(base || '/', specifier);
      const candidate = ['', '.js', '/index.js'].map(suffix => `${joined.replace(/\.ts$/, '.js')}${suffix}`).find(item => item in outputs);
      if (!candidate) throw new Error(`cannot resolve ${specifier} from ${path}`);
      return load(candidate);
    };
    platformModuleRealm(platform).evaluateModule(text, { exports: module.exports, require, module });
    return module.exports;
  };
  const exported = new Map<string,Record<string,unknown>>();
  return (path: string) => {
    const absolute = path.startsWith('/') ? path : `${root}/${path}`;
    const emitted = `${root}/.natlang/build/${absolute.slice(root.length + 1)}`.replace(/\.m?ts$/, '.js');
    const prior=exported.get(absolute);if(prior)return prior;
    const raw=load(emitted),source=project.files[absolute]??project.files[absolute.slice(root.length+1)];
    if(source===undefined)return raw;
    const record=parseModule(absolute,source,{},virtualSourceFiles(project.files,root),{compiled:true});
    const bound=Object.create(Object.getPrototypeOf(raw)),wrappers=new Map<string,WeakMap<Function,Function>>();
    const wrap=(key:string,value:unknown)=>{
      if(typeof value!=='function')return value;
      let cache=wrappers.get(key);if(!cache){cache=new WeakMap();wrappers.set(key,cache);}
      let wrapped=cache.get(value);if(!wrapped){wrapped=new Proxy(value,{apply(fn,thisArg,args){return invokeNativeEntry(record,key,args,()=>Reflect.apply(fn,thisArg,args));}});cache.set(value,wrapped);}
      return wrapped;
    };
    for(const key of Reflect.ownKeys(raw)){
      const descriptor=Object.getOwnPropertyDescriptor(raw,key)!;
      if(typeof key==='string'){
        if('value' in descriptor)descriptor.value=wrap(key,descriptor.value);
        else if(descriptor.get){const get=descriptor.get;descriptor.get=function(){return wrap(key,Reflect.apply(get,this,[]));};}
      }
      Object.defineProperty(bound,key,descriptor);
    }
    exported.set(absolute,bound);return bound;
  };
}

/** Default local composition; hosts can offload only the pure first stage. */
export function compileVirtualProject(project: VirtualProject, runtime: Record<string, unknown>, options: VirtualProjectOptions): CompiledProject {
  const product = compileVirtualProjectProduct(project, {
    ...options,
    target: options.target ?? (moduleTarget(options.platform) === 'await-restoring' ? 'browser' : 'node'),
  });
  return instantiateVirtualProjectProduct(product, runtime, options);
}
