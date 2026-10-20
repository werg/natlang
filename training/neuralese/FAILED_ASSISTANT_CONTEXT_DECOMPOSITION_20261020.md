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
