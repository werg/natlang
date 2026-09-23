export default async function commit(head: string, edits: Edit[], removes: string[], effect_ids: string[], message: string): Promise<CommitResult> {
return await host.research.commit(head, edits, removes, effect_ids, message);
}
