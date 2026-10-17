# Prompt use of admitted data with comprehensive coverage

The supervising policy is: give useful newly admitted cohorts real training
promptly, continue older obligations, and retain earned per-row exposure.
The October 17 saved LFM cursor showed 44,019 old positive-priority rows ahead
of 15,770 and 4,299 new rows with no visits. Source registration alone had not
achieved the intended adaptive training behavior.

The declared `urgent_cohort_fairness` setting selects the policy below. Its
default is false for historical runtimes; production continuations enable it
explicitly in their recipe and checkpoint options.

1. Identify cohorts from immutable source declarations. An explicit positive
   priority declaration groups its child source; otherwise descend through the
   existing source tree. Keep graph bytes, bindings and physical order intact.
2. Pack whole graphs within each cohort. Keep the existing communication
   foundation ratio and priority-zero debt lane.
3. Among pending positive-priority cohorts, select the one with least completed
   indexed graph token exposure. Compute service from saved per-row visit counts
   multiplied by indexed forward tokens. Use declared priority and source
   order to break ties. Add each planned batch's token cost before selecting
   the next one. These widths precede optional causal-prefix sharing and are
   service units, not measured GPU work. Finite cohorts leave the queue when
   their obligations finish.
4. On append or resume, recompute from actual saved credits. Never invent visits,
   reset history or use a global-step modulo that aliases the foundation cycle.
5. Stream each cohort monotonically through existing authenticated readers.
   Check row IDs and all existing content/count/stat checks; close every reader
   on completion or interruption. Mark each visit's final batch after ordering.

This natural-language policy extends the existing training recipe. Crisp
helpers implement the exact scheduling, heap arithmetic, source binding and
reader plumbing. There is one shared implementation for both students; no
model-facing prompts or token semantics change. Complete-coverage, resume,
append, packing and compressed-reader tests verify the mechanism. Cloud
adoption requires an explicit recipe continuation and observed real visits;
local implementation is not an adoption or exposure claim.

Local validation on October 17: 27 focused scheduler/foundation/compressed-reader
tests passed, followed by 139 integrated SFT, resident, continuation, recipe and
drained-handoff tests. The seven cohort tests also passed after adding explicit
unfinished-recipe tuning coverage. The quick main check passed its 22 checks
and TypeScript validation. Independent cloud-owner source review found no
blocking gap and clarified that indexed token service precedes prefix sharing.
Actual production deployment and observed cohort visits remain separate work.
