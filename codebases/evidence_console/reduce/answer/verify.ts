export default function verify(passages: Passage[], draft: Draft, collection_revision: string): EvidenceAnswer {
return host.evidence.verify(passages, draft, collection_revision);
}
