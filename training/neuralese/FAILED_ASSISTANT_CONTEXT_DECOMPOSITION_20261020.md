# Retain failed actions as context, not positive assistant targets

`selectUsefulContinuation(actualFeedback, admittedSources) -> SourceView` is the
existing adaptive training policy: preserve full trajectories, learn recovery,
and propagate useful corrections to both students. This change is exact static
source-view/mask plumbing; it changes no executor prompt or runtime semantics.

1. Authenticate the original structured context/target/tool input with the
   existing no-embedded-chat-boundary guard. Preserve all original raw artifacts,
   source/split/teacher/runtime identities and the exact sampled assistant target.
2. Reuse the existing `chat_roles` and `failed_assistant_mask` law used by whole
   trajectories. Only a prior assistant turn followed by explicit runtime failure
   feedback is excluded from positive CE. Its tokens remain visible causal input.
3. Preserve every other existing weight and zero, including successful assistant
   history, system/user context and tool/error feedback. The current target suffix
   never changes. Keep partial/guided all-history exclusion semantics intact.
4. Give the corrected static source view an explicit immutable identity; do not
   rewrite admitted old rows, completed exposure, optimizer/RNG or old manifests.
   Existing balanced-mass CE honors these new zeros but cannot infer missing masks
   on historical source rows from its suffix/source coefficient alone.
5. Verify actual admitted TRAIN token coordinates and shared both-model reference
   tests, plus source identity/guard/cache separation and continuation compatibility.
   Protected evaluation is diagnosis only and supplies no training answers.

Actual bounded evidence: all 12 already-visited admitted TRAIN success-target
rows had failed assistant history; 1,706 tokens remained positive at weight .05.
The repaired converter, authenticated against the exact original raw captures,
zeroed exactly those 1,706 positions and changed no other weights or target token.
This is a supervision bug; it does not establish the cause of every held failure.

Static CLI: `--failed-assistant-history-policy mask-runtime-failures` creates
explicit `:assistant-history:mask-runtime-failures` view IDs and a receipt field.
Strict admission requires the actual tokenizer and reconstructs the full source
render and mask. Legacy/default IDs and masks remain unchanged. Full-source
repair must not merely append duplicates while continuing incorrect old weights.
The next production provision is an immutable source-bound mask manifest through
the existing CE continuation hook, preserving the old raw row IDs, binding,
optimizer/RNG and exposure while changing only future failed-token supervision.

## Same-ID production continuation (next provision)

1. Prepare an immutable manifest only for rows with positively weighted failed
   assistant tokens. Bind original source SHA, complete original row digest,
   train split/groups/tokenizer, exact zero spans and actual failure evidence.
   Teacher-derived rows require the original raw capture and structured-input
   guard; declared whole-trajectory rows bind their explicit failed-weight-zero
   source intent. Missing custody is actionable preparation failure, not approval.
2. Package a small exact tokenizer snapshot with the manifest. Runtime verifies
   its files and fingerprint, derives roles and failures from actual old row
   tokens using the SAME helper, and rejects any mismatch in spans/evidence.
   No model weights, generation or raw teacher bulk data are needed by learners.
3. Extend the existing CE policy with bound mask-manifest inputs. On TRAIN only,
   zero those verified positions BEFORE existing balanced-mass weighting; retain
   all other raw weights/zeros. Do not touch held telemetry or policy objectives.
4. Keep source/row/parser/index identities and exposure unchanged. Existing CE
   objective continuation records the new policy binding while preserving every
   optimizer/RNG and credit. Source dependency closure includes manifest and
   decoder assets; adding or removing a manifest is explicit recipe work.

The executable producer is `scripts/neuralese_data/build_native_failed_history_masks.py`.
Pass explicit admitted TRAIN graph leaves with `--source`, the exact original
model tokenizer with `--tokenizer`, and a fresh `--out` directory. Optional
`--capture` relocates original raw collector files by their verified SHA. The
producer streams graphs and raw captures, retaining compact mask records only;
all tokenizer assets are fsynced before the final durable manifest. Its source
scope and counts are explicit; a selected cohort is not a census of all training.

To attach a prepared manifest without reparsing an existing million-row recipe,
use `build_native_ce_supervision_policy.py --prior-policy EXISTING_POLICY
--failed-history-mask-manifest MANIFEST --out NEW_POLICY`. This preserves all
existing train/held source declarations; it never grants admission to held,
withdrawn or previously excluded rows. The existing recipe CE continuation
selects the new bound policy and retains optimizer/RNG/exposure history. Provider
closure uses `ce_policy_dependencies` to ferry the exact manifest and small
tokenizer snapshot before any learner reads the policy. Original raw captures
remain reusable source custody but need not be ferried to the learner.

The failure law binds the exact shared `text_supervision.py` helper file and
failure patterns. This conservative binding deliberately rejects a different
helper revision; a later helper change requires an explicit new mask artifact,
not an intrinsic source-index reparse or silently changed old mask semantics.
Fresh corrected static views and same-ID manifests are separate provisions:
new conversions use the reviewed mask flag, while affected old rows receive
only a declared future objective correction. Neither provision proves a held
behavioral improvement before an actual subsequent evaluation.

Preparation reuses one exact tokenizer fingerprint per owned producer invocation
and checks every row against it, then checks both the unchanged tokenizer and
saved snapshot again before publication. Each graph's role/failure derivation
runs once. The pure record serializer grants no proof or training eligibility:
the learner still independently rederives every claimed span from bound tokens.
This avoids repeated full vocabulary hashing and duplicated long-context scans.

Consumer preparation verifies every manifest and every declared decoder asset,
then canonicalizes identical `(asset inventory, tokenizer fingerprint)` decoder
bindings within that policy to the first verified snapshot. The existing bounded
decoder cache therefore loads one tokenizer for equivalent snapshot copies,
instead of churning when a production policy combines several source families.
Distinct content remains distinct, and a damaged later copy still fails before
the shared cache is used. This changes no masks, row identities or objective law.

Literal control-token spellings in an assistant target do not become context
messages. Ordinary failure masks derive only from the authenticated prefix before
the saved suffix boundary. When the legacy full-target role check encounters a
literal marker, the producer must reconstruct that exact captured full rendering
with the existing teacher renderer and declared history profile, binding the full
token sequence, decoded text and suffix coordinate. The consumer checks that
compact rendering witness against the unchanged original row. No target token is
removed, no source is skipped, and whole-trajectory masking remains unchanged.
Rendering reconstruction is needed only for this exceptional target geometry;
existing role-valid manifests retain their exact spans and evidence.

Already admitted native decision renderings can lack the later teacher-source
field while retaining their original approved native records and registered
renderer receipts. For those explicit inputs, reuse the same producer and mask
consumer. Verify the original receipt's exact native JSON file, record identity,
approved native admission, TRAIN split/groups, template/history policy and the
complete shared gold rendering (tokens, text and target boundary). The receipt's
corpus ID and declared full/program-header view must exactly reproduce the
native-view-qualified row identity. Derive masks only after those proofs; never
infer approval from context attestations, source names or a feedback substring.
This is an objective correction to the existing immutable row, not new data or
new native/whole-graph admission. Other source families still need their own
existing custody and remain explicitly pending until verified.

## Resumable preparation and custody census (2026-10-20)

The natural-language preparation service derives only authenticated TRAIN-row
candidates, then closes original capture custody. A failed capture must not
throw away completed derivation or obscure other custody errors. The crisp
mechanism stores one durable, checksum-bound NON-ADMITTED work file per exact
source SHA, decoder fingerprint and derivation/mask law. Each candidate retains
its original row SHA; normal admission and independent consumer rederivation
remain mandatory. No candidate-work schema is accepted by the learner. Source
bytes are rechecked on resume; changed decoder/law/source selects fresh work.
Capture resolution visits every requested row and records every failed source,
row and turn in a durable non-admitted report, then fails before writing a
normal manifest. A repaired capture resumes saved derivation; it never grants
credit or weakens capture verification.

Candidate work version 2 scopes the existing AST source-law verifier to the
actual derive_context_mask and mask_record callables, plus the complete shared
text_supervision failure law and decoder fingerprint. Capture-custody changes
always rerun without invalidating unchanged derivation. An explicitly supplied
preceding module may locate version-1 work only after its actual callable law
is proved identical; different mask code is rejected. No old cache is relabeled
as admitted, and original consumer rederivation remains required.
