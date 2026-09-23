import type { FileIdentity, RepoSnapshot, SearchHit, SearchResult, Patch, Check, Validation, ChangedFile, MigrationReport } from "../types.js";
import { host } from "natlang:runtime";

export default function report(revision: string, checks: Validation): MigrationReport {
return host.repository.report(revision, checks);
}
