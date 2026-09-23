import type { State, Event, Entry, Hit, Edit, Proposal, CommitResult, CandidateResult, Branch, Receipt, View } from "../types.js";
import { host } from "natlang:runtime";

export default async function affected(head: string, changed: string[]): Promise<string[]> {
return await host.research.affected(head, changed);
}
