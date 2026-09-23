import type { MergeProfile, WikiBlock, WikiPage, WikiUpdate, Conflict, PreparedMerge, MergeDraft, MergeReport, CellResult } from "../types.js";

export default function reject(base: WikiPage, prepared: PreparedMerge): MergeReport {
return { status: 'rejected', page: base, detail: prepared.detail };
}
