import type { State, Event, Entry, Hit, Edit, Proposal, CommitResult, CandidateResult, Branch, Receipt, View } from "../types.js";
import { host } from "natlang:runtime";

export default async function native_search(id: string, query: string): Promise<string> {
return await host.research.nativeSearch(id, query);
}
