export type FileIdentity = { path: string, sha256: string, lines: number };
export type RepoSnapshot = { revision: string, files: FileIdentity[] };
export type SearchHit = { path: string, offset: number, line: number, excerpt: string };
export type SearchResult = { revision: string, hits: SearchHit[] };
export type Patch = { path: string, old: string, new: string };
export type Check = { id: string, status: "passed" | "failed", output: string, output_bytes: number, truncated: boolean, detail?: string };
export type Validation = { revision: string, status: "passed" | "failed", checks: Check[] };
