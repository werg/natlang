import type { File, Passage, Claim, Section, Outline, Document, PublishCheck, PublishReport } from "../types.js";

export default function reject(target: string, checked: PublishCheck): PublishReport {
return { status: 'rejected', target: target, revision: '',
  markdown_sha256: '', html_sha256: '', detail: checked.detail };
}
