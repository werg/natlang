import type { FileIdentity, RepoSnapshot, SearchHit, SearchResult, Patch, Check, Validation, ChangedFile, MigrationReport } from "../types.js";
import { host } from "natlang:runtime";

export default async function validate(revision: string): Promise<Validation> {
return await host.repository.validate(revision);
}
