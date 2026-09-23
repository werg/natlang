import type { FileIdentity, RepoSnapshot, SearchHit, SearchResult, Patch, Check, Validation, ChangedFile, MigrationReport } from "../types.js";
import { host } from "natlang:runtime";

export default function search(query: string, revision: string): SearchResult {
return host.repository.search(query, revision);
}
