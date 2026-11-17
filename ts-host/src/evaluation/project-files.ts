/** Evaluation builds application source; neighboring conventional tests have their own host/test dependencies. */
import { nodeProjectFiles } from '../compiler/node-project.js';
import type { ProjectFiles } from '../compiler/project.js';
import { join } from 'node:path';

const conventionalTest = (path: string) => {
  const parts = path.split(/[\\/]/);
  return parts.includes('__tests__') || /\.(?:test|spec)\./.test(parts.at(-1) ?? '');
};

/** Filter discovery only. An application that actually imports a test file still reads and checks that source. */
export const evaluationProjectFiles: ProjectFiles = {
  ...nodeProjectFiles,
  list: directory => nodeProjectFiles.list(directory).filter(name => name !== '__tests__' &&
    (nodeProjectFiles.isDirectory(join(directory, name)) || !conventionalTest(name))),
  compiler: { ...nodeProjectFiles.compiler!,
    readDirectory: (...args) => nodeProjectFiles.compiler!.readDirectory!(...args).filter(path => !conventionalTest(path)) },
};
