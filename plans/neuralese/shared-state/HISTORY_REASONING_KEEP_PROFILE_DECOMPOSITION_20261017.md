# History-reasoning keep profile: decomposition

## Goal and boundary

LFM's current `last_turn_only` template removes reasoning from prior assistant
turns in a tool trajectory. The native graph builder then creates full-prefix
auxiliary graphs to recover those targets. Mellum's `keep` template already
renders those turns in the complete trajectory. The opt-in profile will let a
recipe select the same `keep` rendering for LFM's training and serving views,
so every original assistant reasoning and reply target can be supervised once
in the whole-trajectory graph. This changes model-facing text and prompt
rendering. The existing backbone default remains unchanged unless a persisted
model/recipe config explicitly selects the profile.

No source rows, split assignments, quality decisions, or target turns are
removed. Existing graph receipts keep their original rendering identity.
Changing an untrained conversion to the new rendering creates an explicit new
view with its own binding; it does not relabel old rows or exposure.

## Components

| Part | Representation | Reason |
|---|---|---|
| Profile choice and validation | One shared Python helper with `backbone-default` and `keep` declarations | Exact selection and fail-closed policy validation; no model decision is needed here. |
| Recipe/model declaration | Typed recipe parameter persisted in `model_config` and checkpoint manifests | Exact durable lineage binding; serving must recover the same choice. |
| Tokenizer binding | Existing `bind_history_reasoning` template adapter, extended to accept the persisted declaration | Keeps one shared renderer for data, evaluation, and inference. |
| Native graph conversion | Existing whole-session rendering and target-mask compiler | Preserves the full source trajectory and each target once; no replay graphs are needed for reasoning made visible by the template. |
| Serving and actor views | Existing `Engine` tokenizer binding from checkpoint `model_config` | Ensures generation prompts use the training rendering policy. |
| Provenance and qualification | Existing rendering configuration/fingerprint, graph receipts, and live executor comparison | Makes the selected view distinguishable and requires evidence before adoption. |

## Data flow

1. Resolve the profile from an explicit recipe/model setting; otherwise use the
   declared backbone default (`lfm2: last_turn_only`, `mellum: keep`). Reject
   unknown values and conflicting rebinds.
2. Bind the tokenizer's chat-template switches and run the existing rendering
   probe. Record profile, backbone, template arguments, and probe result.
3. Render a complete source trajectory with the bound tokenizer. Build the
   usual whole-trajectory graph and masks from source targets. The source
   target-ID set must match the original complete trajectory exactly.
4. Keep existing auxiliary-reasoning behavior for the default profile. Under
   `keep`, the whole-template reasoning-coverage proof should show every
   reasoning target present; auxiliary rows are then naturally absent. This is
   verified from emitted coverage, not assumed from the setting.
5. Persist the same profile in checkpoint model config and use it when the
   serving `Engine` binds its tokenizer. A missing setting follows the old
   default, preserving legacy behavior.
   Keep the established four-field native wire identity unchanged. The exact
   profile is attested separately by checkpoint model config and a tokenizer
   sidecar bound to checkpoint SHA, tokenizer fingerprint, and sidecar SHA;
   raw and shared-semantic RL controls carry those expected bindings into
   learner restore. Explicit profiles require the sidecar, while older default
   checkpoints retain their backbone-declared restore path.
6. Before production adoption, compare both profile variants on the live
   executor using the standing 48-sample measurement protocol. Review task
   behavior, token/target counts, prompt fingerprints, and source-target
   conservation. No live configuration is changed by this implementation.

## Exact checks

- Defaults remain byte-for-byte selected as before for existing LFM and Mellum
  configs.
- LFM `keep` passes the template probe and has all source reasoning ordinals
  present in the whole view; every expected assistant target ID occurs exactly
  once in the target mask.
- Mellum `keep` remains unchanged and passes the same shared assertions.
- An explicit unknown or conflicting profile fails closed.
- Training conversion, task-evaluation Engine, checkpoint restore, and
  inference Engine report the same persisted profile/rendering fingerprint.
- Old conversion receipts remain bound to their old profile; a conversion
  under the new profile receives a distinct rendering identity.
- The live-sample comparison is required before changing a production recipe
  or actor configuration.

## Live measurement recorded; review remains

The live executor comparison completed on 2026-10-18 using the same pinned LFM
weights for 48 baseline and 48 KEEP cases. Exact paired outcomes and limitations
are recorded in `plans/MODEL_FACING_CHANGES.md` and the bound run receipt. The
profile remains opt-in until an independent review checks the integrated
training, serving, conversion, and recipe wiring. Measurement does not itself
change an active trainer or actor.
