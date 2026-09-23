import type { State, Event, Entry, Hit, Edit, Proposal, CommitResult, CandidateResult, Branch, Receipt, View } from "../types.js";
import { host } from "natlang:runtime";

export default async function activate(head: string, candidate: string): Promise<CommitResult> {
return await host.research.activate(head, candidate);
}
