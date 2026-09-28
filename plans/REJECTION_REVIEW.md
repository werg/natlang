# Rejection review — 2026-09-28

## Coverage and evidence

The first audit covers every raw saved result under `runs`: **4,434 results in 76 batches**, including historical
student probes, Bonsai, Luna, folder batches and handoffs. It finds **907 task rejections**, **1,346 admission
rejections** and **585 collector error files without a result at their index**. These populations overlap; do not sum
task and admission rejections. Results saved during the live Bonsai queue increase later snapshot counts.

Run the reusable audit from the repository root:

```bash
node --max-old-space-size=3000 ts-host/scripts/audit-rejections.mjs runs/rejection-audit-20260928/current-policy runs
```

The report has a compact evidence record for every rejected row, diagnostics/actions, actual and expected answers,
admission reasons, and a separate error ledger. It scans job results once, rather than duplicated or incomplete range
exports. Automated review routes distinguish where further review belongs; they are not claims of proven root cause.
The detailed trace investigations focused on recurring patterns in the current generation batches.

## Confirmed system problems and changes

| Problem | Evidence | Decision |
| --- | --- | --- |
| Retired exercise expects a compiler refusal that no longer exists | Seven Bonsai handoffs replay with the same correct answer; only seeded prerequisite fails | Block retired exercises throughout generation/admission; preserve raw exclusions |
| Inline children receive delegation examples but runtime forbids delegation | 402 histories in the current-policy audit contain the blanket-ban error; thousands of failed actions in current rejected handoffs | Replace blanket ban with user's three-layer rule and depth-specific prompt/help |
| An inherited callable namespace is also injected as a capture | Handoff 65 reproduces `Injected binding "facts" is duplicated` under the former current runtime; 91 histories contain this fault | Inherit the callable namespace once; retain ordinary live captures |
| Different generated functions collide by source offsets | New three-layer regression exposed false recursion between distinct eval functions | Include invocation identity and source text in eval function identity |
| Seeded handoff reinjects the failure the teacher should replace | Seeded handoff regression and original rejected-action replay | Allow exact planted-action replacement; retain rejected-side failure proof |
| Correct delegation rejected because seed suggests direct computation | First audit has 47 task-accepted rows carrying the optional-delegation rejection | Record a note instead of rejecting this choice; direct answers also remain valid |
| Preference builder excludes an admitted direct chosen answer | `judged_directly` note was treated as rejection | Accept either route, with admission and rejected-side replay still required |
| Common finite-loop diagnostic lacks an executable pagination shape | Relational/commaqa evals repeatedly call `.pages()` in the loop condition | Explain computing the bound once, with a concrete finite loop example |

### Final delegation policy

- An existing instruction function from a `.nl` file is a root with a fresh budget.
- It may have three active ad hoc layers below it. Layer three cannot create a fourth.
- At layer three, system prompt, opening, offered tools and built-in help stop suggesting ad hoc delegation.
- Inline `nl`, Python `nl`, and `delegate` share this count; sibling branches are independent.
- Named file functions can be called from the final layer and start their own budget. Existing actual-function
  recursion checks and the collector's 128-request/600-second case bounds still apply.

The policy was explicitly steered by the user. It supersedes both the former blanket ban and the temporary proposal
to carry the budget through named file calls. Compatibility uses `execution_policy_version: 2`; obsolete histories
are excluded from training rather than rewritten. The updated spec is [SPEC.md](../spec/SPEC.md).

## Remaining rejection categories

| Category | Findings and next handling |
| --- | --- |
| FOLIO/ANLI and other reasoning | Current handoff failures are dominated by FOLIO (112 of the 166 rejected Luna handoffs). Removing tool failures helps execution but does not establish logical correctness. Keep exact-answer failures excluded and use separate measured retries. |
| Argument/signature mistakes | Some saved inline functions receive `input` while code assumes `item`; others expect an item but receive `{ item, premiseText }`. Opening signatures and actual arguments must remain visible; do not silently unpack arbitrary user objects. |
| Partial file edits/extraction | Folder failures include unchanged/edit-incomplete files, missed clauses, malformed CSV, and wrong returned counts despite passing file thresholds. Preserve both file and returned-value contracts; inspect positive-item recall rather than overall agreement alone. |
| Evidence checks | Three current Hotpot folder runs have accepted answers but missing required support observations. One CUAD run lacks child evidence. Keep excluded pending a general evidence-policy review, with underlying articles/clauses retained. |
| Source/oracle edge cases | An exact oracle rejects `14 surgeries` against `14`; other short answers include extra location wording or a weekday instead of a requested date. Review task wording and oracle normalization explicitly; do not automatically treat every nearby string as correct. CoEdIT fragments may already satisfy their edit instruction. |
| Technique and history rules | Successful runs can still miss required iteration/helper/evidence behavior, use brittle regex judgments, dump histories or make unwarranted edits. These are declared curriculum checks, distinct from the removed optional-delegation penalty. Further changes need a specific measured policy decision. |
| Collection failures | 553 missing-package errors came from the previously repaired runner dependency links; 17 context limits, 9 rate limits, 5 transport timeouts and one malformed tool response remain in historical evidence. They are not negative reasoning examples. Luna stays off; bounded Bonsai recovery owns unfinished eligible cases. |
| Large cases | CUAD/folder runs can consume hundreds of child turns. Request and wall-clock budgets bound collection; timeout cases remain deferred, not accepted and not automatically labeled bad reasoning. File-surface comparison remains incomplete, so its default is unchanged. |

## Validation and live probes

Node and browser builds pass; 123 focused regressions and 14 additional compiler/folder/migration regressions pass.
Browser type checks and the audit-script regression pass as well (138 distinct focused tests).
The interpreter rerun also verifies identical source text in successive generated layers. Tests check three allowed
layers, fourth-layer refusal, the final prompt/help, file-root resets, inherited namespace calls and admission of both
correct direct and delegated answers.

Bonsai runs `queue-v3.jsonl` on frozen `runtime-v4`: six reviewed probes in separate `system-fix.jobs`, followed by
the existing filtered recovery queue. They test ANLI, FOLIO children, captured facts and relational pagination.
The first ANLI probe completes in 71 seconds without the removed ban, but still answers one item wrongly; it remains
rejected. Further probes are ongoing. No Luna workers or model training were started.

The current-policy snapshot of 4,439 results contains **1,983 admitted direct runs, 559 admitted inline runs and 212
admitted runs using only named child helpers**. These are raw-run counts across batches, not deduplicated final
training examples. The eventual data build must still enforce source grouping, split provenance and decision-level
admission. The review intentionally preserves obsolete traces in raw evidence but excludes them from training.
