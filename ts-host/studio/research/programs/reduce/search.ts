import type { State, Event, Entry, Hit, Edit, Proposal, CommitResult, CandidateResult, Branch, Receipt, View } from "../types.js";
import { host } from "natlang:runtime";

export default async function search(head: string, query: string): Promise<Hit[]> {
return await host.research.search(head, query);
}
