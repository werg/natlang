export default function reject(base: Document, prepared: Prepared): MergeResult {
return { status: "rejected", text: base.text, applied: [],
         alternatives: prepared.updates.map(u => ({ update_ids: [u.id], proposal: u.text,
                                                        reason: prepared.error })),
         explanation: prepared.error, base_revision: base.revision,
         updates: prepared.updates, presentation: prepared.presentation };
}
