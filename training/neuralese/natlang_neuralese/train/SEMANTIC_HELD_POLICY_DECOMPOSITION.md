# Semantic held supervision isolation

`interpretQuality(historicalHeld, taskOutcomes) -> Continuation` remains shared
natural-language policy. `measureHistoricalHeld` is exact measurement plumbing:
1. Keep the selected held source, rows, original weights and full-coverage checks.
2. Exclude the TRAIN-source CE declaration from held-forward options. Its source
   hashes cannot authorize a different held source, nor confer training credit.
3. Execute the existing shared joint/distill objective, restore model modes and
   RNG, and retain finite/source/coverage checks. No threshold or prompt changes.

Additional declared-held metrics are a separate explicit source-bound extension;
this repair prevents both accidental train-policy reuse and source-coverage
failure when evaluating the full production semantic recipe.
