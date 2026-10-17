# Continue the current optimizer under a reviewed runtime

The native policy is: keep the original bootstrap as historical authority, keep
all current work and earned exposure, and change runtime/recipe only through an
actual later owned drain. A later checkpoint must not become a new bootstrap.

1. Finish or stop the actual current operation and drain its recorded owner and
   children. Keep the exact full optimizer/RNG/source-exposure checkpoint pin.
2. Authenticate that pin against the current published master, or the existing
   native bridge's verified exact request/result for an ahead-of-master partial.
3. Use existing resident retirement and recipe lineage handoff to authorize the
   next code/source/capacity contract, preserving the original recipe lineage.
4. Snapshot the current config, cursor and production state under their existing
   dead-owner leases. Persist the later authority separately from the bootstrap.
5. Rebind only shared invocation metadata. Keep current checkpoint pointers,
   pending packets, stage cursors, selection history and original bootstrap.
6. Resume the existing operation. Its normal trainer restore and publication
   retain optimizer/RNG/exposure and earn any subsequent updates. SFT, preference
   and policy stages use the same existing continuation machinery.

The natural-language policy above is not an alternate trainer or scheduler.
Crisp code verifies exact hashes, existing drain/recipe/bridge proofs, model and
lineage joins, leases and durable compare-and-swap writes. No model-facing text
changes. No model weights, optimizer state or source-policy files change here.

Whole-main target 987c0c26d has an exact 295-module source/arithmetic witness at
`/home/werg/data/dgxops/whole-main-native-source-arithmetic-proof-20261017.json`
(SHA256 44d4fcda1ba410d5163829017ddb4e4e3380e979720536f1f38aaae4eede3206).
Legacy source-v1/IEEE paths retain 4ea arithmetic; 608's opt-in norm storage
functions are exact. This witness is not a capacity certificate. The existing
source-index metadata-revalidation verifier passes without reparsing graphs.

LF9088 (4dd1990f..., 2523899119 bytes) and M12916 (54497c8b...,
48295784337 bytes) are observed parent snapshots, not future handoff authority.
Always bind the actual latest owned full save at transition time. LF's norm
gains/moments are already FP32; do not enable an irrelevant storage upgrade.
M's explicit FP32 norm storage upgrade still requires its fresh actual MAX.
Ordinary LF runtime capacity continuation needs an independently reviewed exact
inventory proof in the existing capacity helper, not a blind certificate flip.

Current 140-task admission/control code is shared main code. D24's staged fit
closure is 12 files/204355015 bytes with 11988 LF and 12054 M complete pairs;
425/359 remain pending. All capacity, admission and exposure claims remain false
until their real objective qualification and recipe adoption. These data are not
silently admitted by this runtime transition. Preserve all older data deficits.

Focused checks cover real process drains, stale/wrong parent and lineage
negatives, source/certificate/runtime proof validation, partial operation joins,
idempotent interrupted publication and exact bootstrap/cursor preservation.

## Executable transition

After the existing supervisor captures/drains the actual latest owner and the
existing `prepare_drained_recipe_handoff` commits its reviewed recipe/code plan,
create the separate authority with the shared CLI (all arguments are exact
existing artifact paths):

```
python -m natlang_neuralese.rl.native_runtime_handoff \
  --bootstrap ORIGINAL_HANDOFF --handoff LATER_HANDOFF \
  --recipe-handoff LATER_RECIPE_HANDOFF --config STUDENT/config.json \
  --out LATER_RUNTIME_AUTHORITY.json
```

For an actual ahead-of-master partial, also pass `--operation-request` and
`--operation-result`. The existing bridge verifier must accept its fetched full
checkpoint and artifact inventory before snapshots can be committed. Historical
verification uses exact immutable request/result/state snapshots and the actual
independent checkpoint pin, so normal reuse of the live job directory cannot
invalidate the original proof. Verify a proof with `--bootstrap ... --verify ...`.

The next ordinary hot plan retains the ORIGINAL `handoff_receipt` and adds
`runtime_handoff` bound to this new proof. Its existing `commands.seed_command`
hook explicitly selects the reviewed Python package with `PYTHONPATH`; stale old
package prefixes reject. The first application compares the prepared config,
cursor and production-state bindings. It writes invocation metadata and the
existing verification receipt only; it never publishes a checkpoint. A crash
between those two writes resumes idempotently. New current-policy readiness
uses the successor recipe SHA separately from preserved lineage. An old retired
resident receipt remains historical in the untouched cursor; a future legitimate
resident adoption must carry the successor policy in its existing history.

The same shared recipe continuation builds ordinary SFT, offline preference and
RL operation inputs; this mechanism adds no alternative objective or trainer.
Existing objective and fresh policy-capacity gates remain mandatory. New model
execution profiles/storage still use their existing explicit transition checks.

## Focused evidence

`native-later-runtime-authority-both-20261017`: 24 CPU tests passed, 17.54s,
actual peak 1.0GiB. Includes real parent/child process drains and independent pins;
exact bootstrap/cursor/cases/selection preservation; wrong origin/current control
and stale command rejection; partial-operation historical replay; interrupted
verification publication; owner lease exclusion; and actual tiny LF/Mellum full
named optimizers. After the metadata transition, both ports' next update equals
uninterrupted weights and optimizer state bit for bit. This is authority and
continuation coverage, not a production model-capacity or task-quality claim.

A current policy/preference optimizer need not impersonate the last resident SFT
owner. When its top-level objective and receipts differ, the recipe proof may
refer to that separately authenticated historical SFT drain. The new actual drain
still owns the latest full model/optimizer/RNG. Both full checkpoints must name
the same base/architecture; the historical stage's complete canonical frame must
match the frame carried in the latest checkpoint, including credits, sources,
quotas and pending obligations; the current global step cannot regress. This is
an exact metadata join, with no old optimizer restore. The historical source-plan
drain, package and checkpoint remain explicit dependencies of the active proof.
The existing retirement and recipe lineage verifier owns the source-plan commit;
no current.json or owner identity is fabricated for the policy worker.

Expanded integration: `native-later-runtime-full-final-20261017` passed 80 CPU
tests, one skipped, 32.88s/1.2GiB peak, across raw rounds, supervised source resume,
bootstrap cache and existing drained recipe proofs. Both actual ports also pass
bitwise next-update continuation after a genuine later resident append. The
source-plan join has separate policy/preference success and tampered-credit
negatives. Historical partial selected-path resolution is snapshotted during the
verified preparation, so later legitimate offload-symlink reuse does not make a
proof depend on mutable live paths. Quick checks: 22 ratchets and TS passed.

Final partial-alias durability rerun:
`native-later-runtime-final-boundary-20261017`, 24 passed/23.76s/1.1GiB;
`native-later-runtime-quick-sealed-20261017`, 22 ratchets plus TS green.
No production model qualification, cloud adoption, or extra exposure is claimed.

## Bare directly owned continuation

A genuine current owner may have no resident plan. In that case, authenticate
its actual saved current recipe separately from the preserved lineage; do not
construct a resident plan. Resolve the existing production stage from that same
recipe, require its exact source receipts in the current or carried saved SFT
frame, and bind the full current model configuration and runtime inventory.
An actual saved objective may carry the production SFT frame only when that
frame's authenticated recipe declares the identical production stage.

The existing drain, independent full pin, published-current or verified-operation
join, original bootstrap, config/cursor/state snapshots and dead-owner leases
remain mandatory. The crisp verifier adds only this explicit authority variant;
it neither changes training state nor authorizes an unexecuted recipe/profile.
The next invocation uses the authenticated saved recipe and package. Existing
operation snapshots and the original bootstrap survive unchanged. Source data,
optimizer moments, RNG, per-row credits, pending packets and unknown control
fields stay in the original checkpoint/control state. No synthetic resident
plan or new qualification is created.

Bare authority CLI supplies `--bare-recipe ACTUAL_SAVED_RECIPE --package
ACTUAL_EXECUTED_PACKAGE --qualification MEMORY_QUALIFICATION_JSON` in place of
`--recipe-handoff`. Qualification bytes are snapshotted durably before commit;
their passed request/parent must equal the actual saved options, and every check
must bind the same runtime inventory and complete model/profile configuration.
No future runtime or merely declared profile can borrow that evidence. Direct
outputs use the existing current-checkpoint publication under the controller
lease before preparation; genuine bridge partials retain their existing exact
request/result verification. Preparation itself never publishes a checkpoint.

`native-bare-runtime-handoff-final2-3576`: 59 CPU tests passed, 64.38s,
1.5GiB observed peak; includes both actual tiny model optimizers' bitwise next
update, genuine process drains, carried-stage authentication, false/profile/code
qualification rejection, immutable qualification reuse and metadata closure.
`native-bare-runtime-quick-3577`: 22 ratchets and TypeScript check passed.
These checks are control/continuation evidence, not production capacity claims.

A bare objective returning from an older resident SFT owner must detach the old
resident invocation hooks from top-level and generated trainer settings. Keep
its historical cursor/receipts and all row credits, but do not attach the old
installed plan to a different current recipe. Literal stale resident CLI flags
reject. `native-bare-resident-detach-final-3589`:20CPU PASS21.47s including
former-resident→bare objective→actual ordinary argv and literal-hook rejection;
quick22/TS GREEN. This is invocation metadata only, no model/source change.
