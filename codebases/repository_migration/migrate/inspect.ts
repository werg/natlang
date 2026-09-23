import type { FileIdentity, RepoSnapshot, SearchHit, SearchResult, Patch, Check, Validation, ChangedFile, MigrationReport } from "../types.js";
import { host } from "natlang:runtime";

export default function inspect(): RepoSnapshot {
return host.repository.snapshot();
}
