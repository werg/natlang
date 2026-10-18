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
