export type FileIdentity = { path: string, sha256: string, lines: number };
export type RepoSnapshot = { revision: string, files: FileIdentity[] };
export type SearchHit = { path: string, offset: number, line: number, excerpt: string };
export type SearchResult = { revision: string, hits: SearchHit[] };
export type Patch = { path: string, old: string, new: string };
export type Check = { id: string, status: string, output: string,
  output_bytes: number, truncated: boolean, detail?: string };
export type Validation = { revision: string, status: string, checks: Check[] };
export type ChangedFile = { path: string, before_sha256: string, after_sha256: string };
export type MigrationReport = { status: string, base: string, revision: string,
  changed: ChangedFile[], checks: Check[] };
