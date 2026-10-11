# Native turn rendering with authenticated source pieces

## Purpose

Render admitted native teacher turns after recovering exact prompt bodies from
their immutable piece table. This extends the existing native turn renderer;
it does not decide which targets are admitted or change their source labels.

## Stages

1. **Select admitted turns.** Keep the renderer's existing rules for native
   schema, `training_admission.approved`, explicit reservations, protected
   items, held source groups, and whole invocation chains. The data files and
   reservation lists are crisp inputs because their bytes and hashes must be
   exact.
2. **Resolve source pieces.** Match registered sources' exact bytes against
   immutable corpus manifests in `training/neuralese_corpora.json`, using their
   canonical logical paths while checking resolved file bytes. Load a piece
   table only when that manifest lists `native-pieces.jsonl` or `pieces.jsonl`
   beside the source. Equivalent registrations are accepted only when their
   declared piece-table hashes agree. Supplying an exact body is a crisp lookup;
   missing references continue to produce explicit omissions. Source,
   manifest, and piece-table hashes are recorded in the receipt. No undeclared
   files are searched. A caller may pass an explicit `--pieces` table for an
   unregistered source; its exact path, size, and hash are recorded, and
   conflicting bodies fail. No body is authored, normalized, or inferred here.
3. **Render each approved decision.** Use the shared `gold_text_rows` and
   `authenticated_crisp_context_messages` functions for full-prompt and
   program-header rows. Use the same authenticated messages and shared chat
   renderer for whole trajectories. Tokenization, template application, and
   rendering parity are exact operations, so they remain crisp helpers.
4. **Verify and publish receipts.** Preserve current split, group, duplicate,
   parity, and omission accounting. Hashes, file writes, and atomic publication
   are crisp mechanisms; no admission policy is added by this change.

## Unit mapping

| Concern | Implementation | Form and reason |
| --- | --- | --- |
| Admission, split, reservation, and protected-item decisions | Existing renderer selection stages | Crisp checks are retained as exact verifiers of already recorded decisions and explicit split policy; this patch adds no new decision rule. |
| Piece identity and content | Immutable source manifest binding | Crisp lookup and SHA-256 receipt; a registered source table supplies the exact bytes when present. |
| Context hydration | `authenticated_crisp_context_messages` | Existing shared helper, now given the declared piece map so its source-hash checks can resolve references. |
| Full and program-header rows | `gold_text_rows` | Existing shared renderer and exact parity check. |
| Whole-trajectory rows | Existing crisp-message and native chat helpers | Existing shared renderer; context pieces only fill exact source references. |
| Counts, omissions, receipts, and writes | Existing receipt and merge helpers | Crisp accounting and durable file output. |

No policy interface or model-facing wording is changed. The newly rendered
prompt content is copied byte-for-byte from the hash-bound source piece table.
