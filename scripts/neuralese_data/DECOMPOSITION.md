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

## Grounded teacher continuation derivatives (2026-10-15)

A recorded next action is usable imitation data but is not a unique oracle for
an independent future-action question. Preserve that source. For the explicitly
declared recorded-extraction derivative, the existing typed directory reducer
and inline natural-language child remain the program: read every item, retrieve
the exact recorded assistant message supplied as evidence, accumulate answers,
write/read the answer map. The child question explicitly asks for extraction.

`ground_teacher_continuations.py` is the crisp schema transformation: bind the
unchanged source program hash, supply the recorded target as labeled evidence,
update exact item/reference hashes, retain all split/source/protection groups,
and mark the derivative's contract and zero independent-world credit. Exact
reference replay, live teacher collection and downstream admission remain the
existing mechanisms. No source is silently discarded, no failed earlier sample
is relabeled, and supplying source targets grants no training admission.

The v2 derivative uses the existing `json-string-record` file verifier for the
explicit answer-map contract. Its keys and every string character stay exact;
JSON whitespace/key ordering is not a task requirement, while other source
files remain byte-checked. This is a corrected exact verifier of the declared
output, not a new prompt or a relaxation of expected answers. The transformer
requires that known output contract before changing an inherited byte oracle.
Historical v1 programs/results remain bound to their original verifier.

## Paired role spans from actual rendered tokens

The pair converter locates its inserted producer/consumer boundaries in the
shared renderer's actual token offsets. Structured chat rendering can tokenize
adjacent pieces separately; merging their decoded text and tokenizing it again
can produce different, equally valid BPE IDs. The exact verifier preserves the
original IDs, checks full decoding, offset geometry, marker identity, payload
hash and each witnessed read occurrence. Rows without renderer offsets retain
the canonical-tokenization check. No payload, target or admission rule changes.

## Authenticate original pair endpoints before static role normalization

The admitted communication edge selects an exact writer result and every exact
structured reader occurrence. Its policy is to keep the full source turn while
replacing that selected communication by native producer/consumer role spans.
Original runtime capture receipts authenticate original bytes; they cannot be
reused as receipts for a rewritten target or message.

1. Validate original admission, split/groups, writer source and provider/capture
   receipts through the existing gold-text authentication path, before rewriting
   either endpoint. Invalid originals remain omissions, including forged target,
   body, context counts and selected-action witnesses.
2. Preserve an authenticated context rendering input plus the original target,
   with exact original-record/message/target hashes and source attestations.
   A shared crisp rendering helper applies explicit structured replacements only
   after authenticating that original input. It is a deterministic derivation,
   not a new admission or a preview route.
3. Derive only the declared writer target and exact authenticated reader paths.
   Bind each old/new value and resulting context/target to a static-transform
   receipt. Original source receipts remain labeled original-source evidence;
   never claim they authenticate changed bytes. Untouched target/context,
   capture augmentations and protected split/source membership stay intact.
4. Use the same native gold packet renderer, escaping, token offsets and role
   marker materialization as ordinary CE. Preserve ordinary CE targets and
   source records; only the separate paired view is transformed.
5. Verify forgery rejection and both actual tokenizers on previous omissions.
   Final pair admission still requires existing exact occurrence/graph/token
   checks and current protection; successful rendering grants no admission.

Policy is this declared source-to-pair derivation. Authentication, exact value
replacement, rendering and hash/offset receipts are crisp mechanisms shared
with ordinary text conversion; there is no separate relaxed renderer.

## Acquired full-session reasoning coverage (2026-10-16)

The objective is complete source-session supervision while avoiding auxiliary
copies of reasoning the actual serving chat template already exposes and the
whole-trajectory graph already targets. Preserve the entire original session
as the whole view. Add a full-prefix reasoning packet only for an assistant
reasoning message that the pinned, probed serving template omits from that
whole view. The final assistant reasoning remains covered by the whole graph.

1. For each nonempty assistant reasoning field, replace only that field in a
   copy of the conversation with a unique source-bound sentinel. Render the
   complete conversation through the shared serving/native render path without
   tokenizing. The sentinel's exact presence proves that specific message is
   represented; duplicate reasoning text or echoes in other roles cannot count
   as coverage. Reject collisions and repeated sentinel output. Add a
   full-prefix packet only for sentinels absent from the whole rendering.
2. Preserve each auxiliary packet's original prefix, tool schemas, source
   identity, split, and source groups. Its target is the exact original
   reasoning-bearing assistant message, rendered through the existing serving
   packet renderer. Do not shorten context or deduplicate by string matching.
3. Record the declared history policy, per-message sentinel result, and exact
   covered/auxiliary assistant ordinals in each view's provenance and receipt
   counts. The original complete session remains immutable evidence.

This is a source-coverage decision encoded in natural-language policy; the
existing serving probe, native graph renderers, exact source hashes, and split
verifiers are crisp mechanisms. It changes no prompts or serving behavior.

## Acquired full-session durable conversion (2026-10-16)

The complete input is first scanned to finish the source-wide held-group
closure. Conversion then advances in bounded, ordered session shards. Each
committed shard pins the complete input digest and record range, source/export
receipts, held/quality/pending protection proof, current decision-hold file
bindings, tokenizer fingerprints, and the exact renderer/runtime file hashes.
It writes train and held graph frames plus an explicit omission ledger, then
publishes the shard receipt before advancing the durable frontier. A restart
reuses only a contiguous prefix whose binding and every output digest match;
uncommitted shard scratch is rebuilt from its source range. Final train/test
outputs concatenate the verified compressed frames without rewriting graph
rows, and session/graph/omission counts must account for the complete source.

Exact file hashing, shard receipts, and atomic final assembly are crisp
durability mechanisms. Split closure and admission reasons remain explicit
source policy; a shard boundary never changes a session's data or split.

The whole original trajectory supplies `writer_context_records` to the shared
authenticated writer index; only the chosen endpoints become target rows. A
reader may need an earlier approved writer beyond the selected edge endpoints.
Exact duplicate source IDs must agree byte-for-byte; same-split/source eligibility
uses the existing writer rules. For new paired-source literal writes outside the
legacy runtime index, bind the exact approved writer record SHA and unique named
`extract_write_sites` descriptor. Typed block IDs and raw authenticated body hashes
must match that writer; body text alone never joins an edge. Keep raw source
bodies distinct from legacy display wrappers. The unchanged original source
receipts are nested as original-only evidence in the derived graph provenance.
