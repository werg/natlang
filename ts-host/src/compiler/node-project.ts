/** Node adapter for the project compiler: real files, `ts.sys` for libraries and dependencies. */
import ts from 'typescript';
import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { createRequire } from 'node:module';
import { compileProject, type BuildOptions, type BuildResult, type ProjectFiles } from './project.js';

export const nodeProjectFiles: ProjectFiles = {
  isFile: path => existsSync(path) && statSync(path).isFile(),
  isDirectory: path => existsSync(path) && statSync(path).isDirectory(),
  list: dir => readdirSync(dir),
  read: path => readFileSync(path, 'utf8'),
  write: (path, text) => { mkdirSync(dirname(path), { recursive: true }); writeFileSync(path, text); },
  compiler: ts.sys,
};

/** Node's type declarations, shipped with the runtime for projects that do not install them. */
const NODE_TYPE_ROOTS = [dirname(dirname(createRequire(import.meta.url).resolve('@types/node/package.json')))];

export function buildProject(options: Omit<BuildOptions, 'files'> & { files?: ProjectFiles }): BuildResult {
  return compileProject({ runtimeTypeRoots: NODE_TYPE_ROOTS, ...options, project: resolve(options.project), files: options.files ?? nodeProjectFiles,
    ...(options.outDir ? { outDir: resolve(options.outDir) } : {}) });
}

export function checkProject(project: string, options: Omit<BuildOptions, 'project' | 'emit' | 'files'> = {}): BuildResult {
  return buildProject({ ...options, project, emit: false });
}
