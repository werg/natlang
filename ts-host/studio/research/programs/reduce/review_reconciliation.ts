import type { State, Event, Entry, Hit, Edit, Proposal, CommitResult, CandidateResult, Branch, Receipt, View } from "../types.js";
import { host } from "natlang:runtime";

export default async function review_reconciliation(candidates: string[]): Promise<string> {
return await host.research.reviewReconciliation(candidates);
}
