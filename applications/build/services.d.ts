// Typing of the host service for callable-folder TypeScript (build/step/*.ts). Natural-language stages learn it from
// buildDeclaration in index.ts.
declare module 'natlang:services' {
  import type { BuildWorkspace } from './index.ts';
  export const build: Pick<BuildWorkspace, 'implementation' | 'inspect' | 'execute' | 'reuse' | 'mismatch'>;
}
