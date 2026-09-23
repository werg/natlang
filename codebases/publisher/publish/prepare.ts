import type { File, Passage, Claim, Section, Outline, Document, PublishCheck, PublishReport } from "../types.js";
import { host } from "natlang:runtime";

export default async function prepare(document: Document, target: string): Promise<PublishReport> {
return await host.publisher.publish(document, target);
}
