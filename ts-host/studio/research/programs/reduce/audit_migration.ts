import type { State, Event, Entry, Hit, Edit, Proposal, CommitResult, CandidateResult, Branch, Receipt, View } from "../types.js";
import { host } from "natlang:runtime";

export default async function audit_migration(head: string, source: string, receipt: string, spec: string): Promise<string> {
return await host.research.auditMigration(head, source, receipt, spec);
}
