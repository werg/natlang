export default function read(span_ids: string[], collection_revision: string): Passage[] {
return host.publisher.evidence.read(span_ids, collection_revision);
}
