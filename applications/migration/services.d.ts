// Typing of the host service for callable-folder TypeScript (migrate/edit/exact.ts). Natural-language stages learn it
// from repositoryDeclaration in index.ts.
declare module 'natlang:services' {
  import type { RepositoryMigration } from './index.ts';
  export const repository: Pick<RepositoryMigration, 'implementation' | 'count' | 'lines' | 'search' | 'read'>;
}
