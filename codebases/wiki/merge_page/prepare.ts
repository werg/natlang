import type { MergeProfile, WikiBlock, WikiPage, WikiUpdate, Conflict, PreparedMerge, MergeDraft, MergeReport, CellResult } from "../types.js";
import { host } from "natlang:runtime";

export default function prepare(base: WikiPage, updates: WikiUpdate[], profile: MergeProfile): PreparedMerge {
return host.wiki.prepare(base, updates, profile);
}
