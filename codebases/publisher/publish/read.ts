import type { File, Passage, Claim, Section, Outline, Document, PublishCheck, PublishReport } from "../types.js";
import { host } from "natlang:runtime";

export default function read(span_ids: string[], collection_revision: string): Passage[] {
return host.publisher.evidence.read(span_ids, collection_revision);
}
