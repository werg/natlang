import type { File, Passage, Claim, Section, Outline, Document, PublishCheck, PublishReport } from "../types.js";
import { host } from "natlang:runtime";

export default function check(document: Document): PublishCheck {
return host.publisher.check(document);
}
