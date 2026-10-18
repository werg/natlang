# DeepDive reference-answer admission

The remaining D31 QA files contain 1,016 SFT and 2,234 RL questions with
reference answers and empty conversations. They are reusable source material,
not observed tool trajectories or samples from either student. The natlang
unit is a research function that reads a supplied question and returns its
source reference answer. This is ordinary capability supervision; it supplies
no evidence of learned communication or verified research execution.

1. Read both pinned QA parquet partitions through the existing source adapter.
   Keep partition, row number, upstream id and revision in provenance. Reject
   missing question/answer, unexpected nonempty conversations, or template
   markup that the shared record format cannot retain explicitly. Literal
   quoted reasoning/tool-call syntax remains ordinary answer text.
2. Supply the question as the task background and the unchanged answer as the
   assistant target. Reuse the shared trajectory and native conversion paths;
   do not synthesize tools, reasoning, successful execution or on-policy actors.
3. Give QA rows distinct partition-qualified session ids. Share the exact
   question/task groups with D31's existing research trajectories so protected
   question closure also protects these answer views.
4. Deduplicate and close splits against the existing protected/closed records
   before training admission. Keep QA RL as source provenance, not a claim
   that static answer supervision performs reinforcement learning.
5. Render and stage both students with the existing shared native pipeline,
   preserving full pending rows. Count training only after actual adoption
   and per-row exposure. The same question/answer material may subsequently
   seed self-hosted teacher or RL tasks with separately verified execution.

The policy above is natural language. Python only reads parquet, constructs
exact records, checks source schema and preserves identity/grouping. No new
runtime policy or alternate training implementation is needed. The new task
instruction is tracked as model-facing measurement debt until sampled on the
live executor; source conversion alone does not discharge that debt.

Actual source preparation (2026-10-18): both exact pinned QA files yielded
3,250 valid records with no adapter omissions. Joint closure with all 3,016
existing closed D31 trajectory records and protected-bgkit SHA
`585b21e7091adfbc4f851ef35b0b3c6da00f032fffe69a8b98816a567abad7bf`
removed 41 exact duplicates, produced 3,209 new QA records and had zero invalid
records, protected hits or closure violations. Existing trajectory records are
closure inputs only and are excluded from the new output. Bound preparation:
`/home/werg/data/external-expansion/deepdive-reference-qa-v1/preparation.json`.
Native lowering/rendering, admission and actual exposure remain separate work.

All 3,209 items subsequently lowered through the shared native pipeline:
803 cases, 8,024 decisions and zero omissions. The first Docker text-render
attempt exposed an import bootstrap bug: direct script execution did not put
the repository root on Python's module path, so the shared output-lease import
failed before rendering. The common Docker wrapper now supplies the repository
and shared training package roots explicitly. Reuse no failed render receipt;
start a fresh policy-bound rendering with that corrected wrapper.
