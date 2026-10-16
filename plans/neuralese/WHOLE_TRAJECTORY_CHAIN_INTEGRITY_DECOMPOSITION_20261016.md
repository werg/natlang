# Authenticate whole-trajectory coverage and source protection

The admitted-decision renderer must preserve every independently usable answer,
credit a source only when that answer occurs in the actual trajectory, and keep
held source context out of training. These are exact source verifiers, not model
policy or prompt changes.

1. Define one shared exact predicate for an earlier decision occurring in a
   later source history: its complete message prefix matches, the next message
   is an assistant, and that message equals its admitted target (or the existing
   explicit teacher-reasoning rendering of that target).
2. Use the predicate when selecting maximal whole trajectories and their
   members. Different answers to the same prefix remain separate examples.
   Group by source invocation and split; never credit train rows into held rows
   or the reverse. Preserve deterministic ordering and all source hashes.
3. Emit the complete union of factual source groups from every credited member.
   Apply existing held-group protection to that union. Do not invent groups.
4. Before rendering, reject train rows whose own groups are held or whose exact
   history contains a protected decision from the same invocation. Keep valid
   alternate-answer branches; do not treat all shared prefixes as held answers.
   Record the source ID/group explaining each exclusion.
5. Check branch preservation, authentic source-credit sets, split separation,
   group union, and held-context exclusion with real source structures. Compare
   actual pinned-tokenizer renderer/converter output on both students. Audit
   current corpus relevance before requiring any rerender. Do not edit the live
   canonical renderer while Search/OWL is running against its pinned code.

Inspection of real immutable Search/OWL source localized legitimate dictionary
differences to runtime tool-call ID renaming and explicitly supplied stock
reasoning. The shared assistant-binding verifier therefore binds both raw and
historical assistant hashes and ordered call-ID rebindings, allowing no action,
argument, role or reply changes. Source-declared reasoning can appear in history
even when `teacher_reasoning_trained:false`; its presence does not admit it as
a reasoning target. Emit hash-bound per-assistant reasoning holds. Native
conversion zeroes only the actual reasoning body tokens, retaining the entire
transcript, real actions and structural thinking boundaries. LFM's declared
template drops historical reasoning, so its actual masked count can be zero;
Mellum retains it across tool turns and requires the body mask. Keep all views'
identity, deduplication and target statistics consistent with these masks.

Emit per-member group maps and hashes for every whole trajectory, independently
of quality-target holds. Validate the complete group union during native
conversion. Different groups within one same-split chain are legitimate;
authenticate each instead of requiring equality to the terminal record's groups.
Selection blocks use split-local keys. Older completed selection states retain
their explicitly conservative invocation-wide interpretation on resume.

Exact-text deduplication additionally requires the same complete source/group
and mask identity. It may collapse different view IDs of one exact source;
independent source cases remain independent even when a tokenizer renders their
texts identically. This prevents provenance from disappearing at part, shard or
corpus-union merges. Existing held-text protection still compares the text
regardless of source identity or masks.

Qualify new whole-trajectory row IDs with the full tokenizer-independent source
identity digest before corpus union's first ID selection. Retain the prior row
ID as a crosswalk in both the text row and converted graph. This preserves
independent source/mask variants instead of allowing a reused view ID to discard
one before the stronger content deduplication runs.

Evidence so far: independent review corrected group-union, split-local block
and merge-provenance gaps before adoption. Ledger
`whole-chain-qualified-final-20261016` passed 91 focused tests. Actual pinned
tokenizers in `whole-chain-qualified-tokenizers-20261016` retained every token,
protected held context and preserved the alternate-answer branch: LFM57tokens /
0 masked historical-reasoning tokens; Mellum91tokens /8 body tokens masked.
The first revised fixture incorrectly expected LFM to retain history; its failed
assertion led to the architecture-specific check above. These fixtures are
synthetic protocol checks, never generated/admitted training data.

An immutable first Search/OWL shard census (2,500 approved raw candidates, before
final render exclusions) found unchanged1,250 whole documents and zero unmatched
members after the declared transport/reasoning bindings. Earlier raw-dictionary
mismatch counts are retracted as evidence of changed answers. This scoped census
does not prove all existing corpora unaffected. Data owner is auditing actual
retained reasoning in active evidence views before any withdrawal/rerender.
