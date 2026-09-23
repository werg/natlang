import type { MergeProfile, WikiBlock, WikiPage, WikiUpdate, Conflict, PreparedMerge, MergeDraft, MergeReport, CellResult } from "../types.js";
import { host } from "natlang:runtime";

export default function publish(base: WikiPage, prepared: PreparedMerge, draft: MergeDraft, profile: MergeProfile): MergeReport {
return host.wiki.publish(base, prepared, draft, profile);
}
