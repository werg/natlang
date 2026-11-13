# Read-before-code probe

The corpus in `data/teacher/read-before-code-probe.ir.jsonl` has 59 program cases. It uses ordinary eager inputs and directory reducer files. The account and event batches are larger than the 800-character opening-preview limit, so only their first few rows appear there. The incident-note cases require interpreting notes that are hidden beyond that preview. The folder cases require reading a policy or criteria file before writing the implementation.

## Teacher observations

The first eight-case Bonsai pilot covered two cases each from account files, event files, account inputs, and event inputs. All eight were accepted, and all eight read the relevant file or input value before the first implementation `eval`. Seven used one `eval`; one used two. Results: `runs/read-before-code-realistic-pilot.results.jsonl`.

A 30-event incident window tested value paging. The first run was accepted, but the teacher read pages starting at 0, 13, and 26, skipping rows at the page boundaries. The result happened to be correct because its code processed the entire input. The `read_value` response now prints half-open page bounds and an explicit `next start` offset. In the rerun, the teacher read `[0,12)`, `[12,24)`, and `[24,30)` and returned the expected answer. Results: `runs/read-before-code-paged.results.jsonl` and `runs/read-before-code-paged-v2.results.jsonl`.

The first two incident-note pilots both read the complete value. The recovery case was accepted. The other exposed ambiguous sample wording and an oracle boundary that allowed reasonable alternative selections; its original results must not be admitted as training data. A revised version spent over six minutes in a single model response, so that case was removed from the corpus. Its interrupted journal remains under `runs/read-before-code-incidents-v4.jobs/` for diagnosis.

## What the probe establishes

The teacher can inspect content omitted by the opening preview before choosing code. The page interface can also support complete traversal when it gives an exact continuation offset. Account and event filter cases show voluntary reading; their visible requests could still be implemented as generic code without manually inspecting every row. The incident-note cases test semantic interpretation of hidden rows more directly. These small pilots do not establish an accuracy rate for the full 59-case corpus.

Run `node ts-host/scripts/build-read-before-code-probe.mjs` to rebuild the corpus and example sheet. Run `node ts-host/scripts/analyze-read-before-code-probe.mjs RESULTS.jsonl` to inspect acceptance, read order, and page navigation.
