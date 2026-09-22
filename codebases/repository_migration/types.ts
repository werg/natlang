export type FileIdentity = { path: Text, sha256: Text, lines: Num };
export type File = { kind: Text, text?: Text, bytes: Num };
export type RepoSnapshot = { revision: Text, files: FileIdentity[] };
export type SearchHit = { path: Text, offset: Num, line: Num, excerpt: Text };
export type SearchResult = { revision: Text, hits: SearchHit[] };
export type Patch = { path: Text, old: Text, new: Text };
export type Check = { id: Text, status: Text, output: Text,
  output_bytes: Num, truncated: Bool, detail?: Text };
export type Validation = { revision: Text, status: Text, checks: Check[] };
export type ChangedFile = { path: Text, before_sha256: Text, after_sha256: Text };
export type MigrationReport = { status: Text, base: Text, revision: Text,
  changed: ChangedFile[], checks: Check[] };
