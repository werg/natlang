import type { State, Event, Entry, Hit, Edit, Proposal, CommitResult, CandidateResult, Branch, Receipt, View } from "../types.js";
import { host } from "natlang:runtime";

export default async function native_read(id: string, offset: number, length: number): Promise<string> {
return await host.research.nativeRead(id, offset, length);
}
