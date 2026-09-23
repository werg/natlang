export default function read(selected: string[], collection_revision: string): Passage[] {
return host.evidence.read(selected, collection_revision);
}
