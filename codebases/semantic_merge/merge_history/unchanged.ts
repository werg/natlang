export default function unchanged(base: Document, prepared: Prepared): MergeResult {
return { status: "merged", text: base.text, applied: [], alternatives: [],
         explanation: "No updates.", base_revision: base.revision,
         updates: [], presentation: prepared.presentation };
}
