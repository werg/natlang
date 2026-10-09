// Typing of the host services for callable-folder TypeScript.
declare module 'natlang:services' {
  import type { SearchQuery, SearchResult } from './types.ts';
  export const index: { search(query: SearchQuery): Promise<SearchResult> };
}
