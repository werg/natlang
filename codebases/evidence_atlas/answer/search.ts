import type { Hit, SearchResult, Passage, Claim, Draft, EvidenceAnswer } from "../types.js";
import { host } from "natlang:runtime";

export default function search(queries: string[]): SearchResult {
return host.evidence.search(queries);
}
