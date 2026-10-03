# TextWorld completion-certificate training exposure audit

Read-only audit; no corpus, gold, checkpoint, or active runtime was changed.

The active source-clean-v2 checkpoint is step **1860**, cursor **14877**, with **14880** examples processed and target **105311**. This corpus split contains 105,317 train rows and 5,410 held-out rows before the approved exclusion adjustment.

The false-positive reference program intersects the training corpus through one program/source group: `inline-curriculum:textworld:textworld_iterate:game17:playable:hinted` / `textworld:17`. It has **36** rows. Six are already consumed at positions 1679, 1850, 2139, 5744, 7236, 9921; the next is position **20824**, 5,947 rows after the current cursor. Thirty remain.

The saved Luna result `runs/hourly-check-20260929-0125/luna.v16.jobs/000111-1e7f1ef15bfbb852.result.json` (SHA-256 `3e6e0130bf15b5064187afdee0c8d042534cd6d18f4f349c7eb5671159288a02`) is for the same program and source. It finished with exact expected value `quest-54e03507`, `accepted=true`, and `checks.world=true`. Its 38-turn trajectory has 38 ledger entries, including 16 with `world.act` and 3 with `world.certificate`. This is a real sequence of environment actions culminating in the trusted certificate, distinct from the new reference false positive where `iterateOn` fails before acting and code returns the literal target.

**Disposition:** fix the future native completion grader; do not hold or delete these 36 historical rows on this evidence. The active frozen training runtime remains unchanged, so the new grader applies only after a reviewed future runtime rollout. Detailed row IDs, exact shuffled positions and hashes are in [exposure-audit-v1.json](exposure-audit-v1.json).
