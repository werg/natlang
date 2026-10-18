# Exact tool contracts and changed-action recovery

The current train-only service-recovery packet covers reserved binding, missing API,
argument typing and missing-file recovery. Its page service silently slices an
invalid page index. The new profile adds real page-range feedback and result-shape
repair using independent constructed worlds; no held prompts, IDs or answers enter
construction. Existing legacy 512-world and terminal-policy outputs stay unchanged.

1. `summarize.nl` remains the named natural-language recovery stage. Given the
   actual tool/type error, it chooses a corrected action from the declared service
   or result contract, then reads all pages, delegates semantic policy decisions,
   aggregates selected records and writes/reads the complete summary. Errors are
   observations; the prompt never lists invalid forms.
2. `summarize/select_record.nl` remains the typed semantic judgment over a complete
   record and policy. Explicit exclusion takes precedence over waiver or ordinary
   inclusion, and the child returns its boolean directly.
3. Crisp service code owns immutable facts and the exact positive-integer page
   range. One sentence reports the valid range. This outside-world validity check
   is justified by observed recovery failures; it is not an inference budget.
4. The existing deterministic constructor supplies eight fresh worlds across two
   domains and four shapes. Three failure-entry variants per world yield 24 cases:
   page range, reserved service binding and declared result shape. Source IDs and
   facts use a new revision. The actual teacher supplies two trajectories per case
   through the existing guardian/qualifier/feeder, with original packet custody.
5. Exact source oracles retain full facts, all original reference runs, expected
   root values and file effects. Focused checks verify old-output identity, source
   split/group separation, branch coverage and real page-service behavior. Fresh
   frozen-runtime reference qualification precedes any queue adoption.
6. The 48 self-hosted teacher trajectories record the live measurement of the new
   service error and instruction profile. Generation, source qualification,
   decision admission, BOTH conversion, actual source adoption and per-row training
   exposure remain separate receipts. The existing stronger teacher, engine and
   sandbox are reused; no protected case is replayed for training.

This is one optional profile of the existing constructor, not a second generation
or collection mechanism. Both students receive all independently verified full
trajectories through the current shared admission and ordinary-SFT pipeline.

Actual reference validation: the first 24-case packet qualified 16/24. Its eight
typed-result seeds used an unannotated wrong return, which was recorded as a
wrong result rather than a recoverable type diagnostic. That packet and its
verdicts remain immutable. The corrected seed assigns the value to declared
`Summary` before returning, so the existing compiler emits the real type error.
Fresh v2 qualification completed 24/24 with zero model calls at 13:51:42 UTC.
`/home/werg/data/dgxops/teacher-tool-contract24-v2-20261018/qualification/prepared.json`
binds all executed source references, protection, runtime inventory and verdicts.
The regression executes the authored TypeScript and requires its actual semantic
assignment diagnostic, in addition to full independent reference qualification.
Own-teacher 48-trajectory generation remains pending.
