import type { FileIdentity, RepoSnapshot, SearchHit, SearchResult, Patch, Check, Validation, ChangedFile, MigrationReport } from "../types.js";
import { host } from "natlang:runtime";

export default function apply(revision: string, patch: Patch[]): RepoSnapshot {
return host.repository.apply(revision, patch);
}
