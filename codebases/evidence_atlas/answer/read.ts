import type { Hit, SearchResult, Passage, Claim, Draft, EvidenceAnswer } from "../types.js";
import { host } from "natlang:runtime";

export default function read(selected: string[], collection_revision: string): Passage[] {
return host.evidence.read(selected, collection_revision);
}
