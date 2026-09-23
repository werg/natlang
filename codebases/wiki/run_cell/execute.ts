import type { MergeProfile, WikiBlock, WikiPage, WikiUpdate, Conflict, PreparedMerge, MergeDraft, MergeReport, CellResult } from "../types.js";
import { host } from "natlang:runtime";

export default async function execute(block_id: string, input: string): Promise<CellResult> {
return await host.wiki.runCell(block_id, input);
}
