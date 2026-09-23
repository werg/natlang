import type { State, Event, Entry, Hit, Edit, Proposal, CommitResult, CandidateResult, Branch, Receipt, View } from "../types.js";
import { host } from "natlang:runtime";

export default async function propose(base: string, edits: Edit[], removes: string[], effect_ids: string[], message: string): Promise<CandidateResult> {
return await host.research.propose(base, edits, removes, effect_ids, message);
}
