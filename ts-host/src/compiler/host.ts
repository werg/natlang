import ts from 'typescript';
import { INTRINSICS_FILE, INTRINSICS_GLOBAL_DTS, INTRINSICS_MODULE_DTS, STD_MODULE_DTS, STD_MODULE_FILE, SURFACE_MODULE_FILE } from './intrinsics.js';
import { desugarNlCalls } from './nl-call.js';

/** Minimal file access the compiler needs. Node supplies `ts.sys`; browsers supply virtual files. */
export interface CompilerFiles {
  readFile(path: string): string | undefined;
  fileExists(path: string): boolean;
  directoryExists?(path: string): boolean;
  getDirectories?(path: string): string[];
  realpath?(path: string): string;
  readDirectory?(path: string, extensions?: readonly string[], excludes?: readonly string[],
    includes?: readonly string[], depth?: number): string[];
}

/** Returns declaration text for a TypeScript library file such as `lib.es2022.d.ts`. */
export type LibProvider = (fileName: string) => string | undefined;

const VIRTUAL_LIB_DIR = '/__natlang__/lib';
const dirname = (path: string) => { const at = path.replace(/[\\/]+$/, '').lastIndexOf('/'); return at <= 0 ? '/' : path.slice(0, at); };

/**
 * The library declarations a bundle embeds when its platform has no TypeScript install to read them from (the browser and
 * workerd builds define this at build time; scripts/build-browser.mjs, scripts/build-core.mjs). Under Node it is not defined.
 */
declare const __NATLANG_TS_LIBS__: Record<string, string> | undefined;
const embeddedLibs: LibProvider | undefined = typeof __NATLANG_TS_LIBS__ === 'undefined' ? undefined : name => __NATLANG_TS_LIBS__![name];

/** Only immutable libraries can outlive their program. Parse callbacks belong to their options owner. */
type ParseOptions = ts.ScriptTarget | ts.CreateSourceFileOptions;
type SourceCache = Map<string, { text: string; file: ts.SourceFile }>;
const sharedNumericLibraries: SourceCache = new Map();
const sharedOptionLibraries = new WeakMap<ts.CreateSourceFileOptions, { options: ts.CreateSourceFileOptions; files: SourceCache }>();
const MAX_SHARED_LIBRARY_FILES = 128;

function cachedLibrary(fileName: string, text: string, options: ParseOptions): ts.SourceFile {
  let cache: SourceCache;
  if (typeof options === 'number') cache = sharedNumericLibraries;
  else {
    const found = sharedOptionLibraries.get(options);
    const previous = found?.options;
    const unchanged = previous && previous.languageVersion === options.languageVersion &&
      previous.impliedNodeFormat === options.impliedNodeFormat && previous.jsDocParsingMode === options.jsDocParsingMode &&
      previous.setExternalModuleIndicator === options.setExternalModuleIndicator;
    cache = unchanged && found ? found.files : new Map();
    // The public options object is mutable. Replace its old cache rather than retaining each past option variant.
    if (!unchanged) sharedOptionLibraries.set(options, { options: { ...options }, files: cache });
  }
  const key = `${typeof options === 'number' ? options : options.languageVersion}\0${fileName}`;
  const found = cache.get(key);
  if (found && found.text === text) {
    cache.delete(key);
    cache.set(key, found);
    return found.file;
  }
  const file = ts.createSourceFile(fileName, text, options, true);
  cache.delete(key);
  cache.set(key, { text, file });
  if (cache.size > MAX_SHARED_LIBRARY_FILES) cache.delete(cache.keys().next().value!);
  return file;
}

export type NatlangCompilerHostOptions = {
  options: ts.CompilerOptions;
  /** Absolute POSIX paths overlaid on the underlying files. */
  virtual?: ReadonlyMap<string, string>;
  files?: CompilerFiles;
  libs?: LibProvider;
  /** Read `nl` called on literal instructions as the tag form (default true; a host that resolves symbols itself,
   * compiler/host-cell.ts, rewrites only the built-in `nl` and passes false). */
  desugarNl?: boolean;
  currentDirectory?: string;
  /** Reuse immutable library ASTs (default true); user and generated program sources are always owned by this host. */
  shareLibrarySourceFiles?: boolean;
};

/** A compiler host over real or virtual files, always including the natlang intrinsics. */
export function createNatlangCompilerHost(config: NatlangCompilerHostOptions): ts.CompilerHost {
  const virtual = new Map(config.virtual ?? []);
  virtual.set(INTRINSICS_FILE, INTRINSICS_GLOBAL_DTS);
  virtual.set(SURFACE_MODULE_FILE, INTRINSICS_MODULE_DTS);
  virtual.set(STD_MODULE_FILE, STD_MODULE_DTS);
  const files: CompilerFiles | undefined = config.files ?? (ts.sys as CompilerFiles | undefined);
  // Library declarations come from an explicit provider, the bundle's embedded ones (browsers, workerd), or the TypeScript install.
  const system = ts.sys as CompilerFiles | undefined;
  const libs = config.libs ?? embeddedLibs ?? (system ? undefined : () => undefined);
  const libDir = libs ? VIRTUAL_LIB_DIR : dirname(ts.getDefaultLibFilePath(config.options));
  const read = (fileName: string): string | undefined => {
    const direct = virtual.get(fileName);
    if (direct !== undefined) return direct;
    if (libs && fileName.startsWith(`${VIRTUAL_LIB_DIR}/`)) return libs(fileName.slice(VIRTUAL_LIB_DIR.length + 1));
    if (!libs && fileName.startsWith(`${libDir}/`)) return system?.readFile(fileName);
    return files?.readFile(fileName);
  };
  const directories = new Set<string>();
  for (const path of virtual.keys()) {
    let dir = dirname(path);
    while (dir && !directories.has(dir)) { directories.add(dir); const parent = dirname(dir); if (parent === dir) break; dir = parent; }
  }
  return {
    getSourceFile(fileName, languageVersion, _onError, shouldCreateNewSourceFile) {
      const read_ = read(fileName);
      if (read_ === undefined) return undefined;
      // `nl` called on its literal instructions reads as the tag form (compiler/nl-call.ts): every program built here
      // (project builds, callable-folder modules, eval cells, the page) compiles nl(`...`), nl(`...`)(x),
      // nl(`...`(x)) and nl(`...`, x) exactly as the tag form; declarations and libraries are never rewritten.
      // A file that declares its own nl keeps its calls as written (that nl is an ordinary function).
      const text = config.desugarNl === false || /\.d\.[cm]?ts$/.test(fileName) || fileName.startsWith(`${libDir}/`) ||
        /\b(?:function|const|let|var|class)\s+nl\b/.test(read_) ? read_ : desugarNlCalls(read_);
      const intrinsic = fileName === INTRINSICS_FILE || fileName === SURFACE_MODULE_FILE || fileName === STD_MODULE_FILE;
      // Virtual overlays and user package/folder/eval declarations are program state, even under our synthetic prefix.
      const library = !virtual.has(fileName) && fileName.startsWith(`${libDir}/`) && /(?:^|\/)lib\.[^/]+\.d\.ts$/.test(fileName);
      return config.shareLibrarySourceFiles !== false && !shouldCreateNewSourceFile && (intrinsic || library)
        ? cachedLibrary(fileName, text, languageVersion)
        : ts.createSourceFile(fileName, text, languageVersion, true);
    },
    getDefaultLibFileName: options => `${libDir}/${ts.getDefaultLibFileName(options)}`,
    getDefaultLibLocation: () => libDir,
    writeFile: () => { throw new Error('natlang compiler hosts do not write through TypeScript emit'); },
    getCurrentDirectory: () => config.currentDirectory ?? (ts.sys && files === ts.sys ? ts.sys.getCurrentDirectory() : '/'),
    getDirectories: path => files?.getDirectories?.(path) ?? [],
    getCanonicalFileName: name => name,
    useCaseSensitiveFileNames: () => true,
    getNewLine: () => '\n',
    fileExists: fileName => virtual.has(fileName) || (!!libs && fileName.startsWith(`${VIRTUAL_LIB_DIR}/`) &&
      libs(fileName.slice(VIRTUAL_LIB_DIR.length + 1)) !== undefined) ||
      (!libs && fileName.startsWith(`${libDir}/`) && !!system?.fileExists(fileName)) || !!files?.fileExists(fileName),
    readFile: read,
    directoryExists: path => directories.has(path) || path === VIRTUAL_LIB_DIR || !!files?.directoryExists?.(path),
    realpath: path => virtual.has(path) ? path : files?.realpath?.(path) ?? path,
  };
}

/** Compiler options for eval snippets and small virtual programs. */
export const EVAL_COMPILER_OPTIONS: ts.CompilerOptions = {
  target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext, moduleResolution: ts.ModuleResolutionKind.Bundler,
  lib: ['lib.es2022.d.ts'], strict: true, noEmit: true, skipLibCheck: true, types: [],
  allowArbitraryExtensions: true, noUnusedLocals: false, allowJs: false,
};

/** Build a checked program over in-memory files (absolute POSIX paths). */
export function createVirtualProgram(files: Record<string, string>, options: ts.CompilerOptions = EVAL_COMPILER_OPTIONS,
  roots: string[] = Object.keys(files), base?: CompilerFiles): ts.Program {
  const host = createNatlangCompilerHost({ options, virtual: new Map(Object.entries(files)), files: base ?? ts.sys });
  return ts.createProgram({ rootNames: [INTRINSICS_FILE, ...roots], options, host });
}
