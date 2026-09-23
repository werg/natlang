import ts from 'typescript';
import { INTRINSICS_FILE, INTRINSICS_GLOBAL_DTS, INTRINSICS_MODULE_DTS, SURFACE_MODULE_FILE } from './intrinsics.js';

/** Minimal file access the compiler needs. Node supplies `ts.sys`; browsers supply virtual files. */
export interface CompilerFiles {
  readFile(path: string): string | undefined;
  fileExists(path: string): boolean;
  directoryExists?(path: string): boolean;
  getDirectories?(path: string): string[];
  realpath?(path: string): string;
}

/** Returns declaration text for a TypeScript library file such as `lib.es2022.d.ts`. */
export type LibProvider = (fileName: string) => string | undefined;

const VIRTUAL_LIB_DIR = '/__natlang__/lib';
const dirname = (path: string) => { const at = path.replace(/[\\/]+$/, '').lastIndexOf('/'); return at <= 0 ? '/' : path.slice(0, at); };
let defaultLibs: LibProvider | undefined;

/** Browser bundles install embedded library declarations here. */
export function setDefaultLibProvider(provider: LibProvider): void { defaultLibs = provider; }

/** Parsed library and intrinsic files are shared by every program with the same target. */
const sharedSourceFiles = new Map<string, { text: string; file: ts.SourceFile }>();

function cachedSourceFile(fileName: string, text: string, target: ts.ScriptTarget): ts.SourceFile {
  const key = `${target}\0${fileName}`;
  const found = sharedSourceFiles.get(key);
  if (found && found.text === text) return found.file;
  const file = ts.createSourceFile(fileName, text, target, true);
  sharedSourceFiles.set(key, { text, file });
  return file;
}

const isShared = (fileName: string) => fileName.startsWith('/__natlang__/') ||
  /[\\/]typescript[\\/]lib[\\/]lib\.[^\\/]*\.d\.ts$/.test(fileName);

export type NatlangCompilerHostOptions = {
  options: ts.CompilerOptions;
  /** Absolute POSIX paths overlaid on the underlying files. */
  virtual?: ReadonlyMap<string, string>;
  files?: CompilerFiles;
  libs?: LibProvider;
  currentDirectory?: string;
};

/** A compiler host over real or virtual files, always including the natlang intrinsics. */
export function createNatlangCompilerHost(config: NatlangCompilerHostOptions): ts.CompilerHost {
  const virtual = new Map(config.virtual ?? []);
  virtual.set(INTRINSICS_FILE, INTRINSICS_GLOBAL_DTS);
  virtual.set(SURFACE_MODULE_FILE, INTRINSICS_MODULE_DTS);
  const files: CompilerFiles | undefined = config.files ?? (ts.sys as CompilerFiles | undefined);
  // Library declarations come from an explicit provider, the platform default (browsers), or the TypeScript install.
  const system = ts.sys as CompilerFiles | undefined;
  const libs = config.libs ?? defaultLibs ?? (system ? undefined : () => undefined);
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
    getSourceFile(fileName, languageVersion) {
      const text = read(fileName);
      if (text === undefined) return undefined;
      const target = typeof languageVersion === 'number' ? languageVersion : languageVersion.languageVersion;
      return isShared(fileName) ? cachedSourceFile(fileName, text, target) :
        ts.createSourceFile(fileName, text, languageVersion, true);
    },
    getDefaultLibFileName: options => `${libDir}/${ts.getDefaultLibFileName(options)}`,
    getDefaultLibLocation: () => libDir,
    writeFile: () => { throw new Error('natlang compiler hosts do not write through TypeScript emit'); },
    getCurrentDirectory: () => config.currentDirectory ?? (files === ts.sys ? ts.sys.getCurrentDirectory() : '/'),
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
