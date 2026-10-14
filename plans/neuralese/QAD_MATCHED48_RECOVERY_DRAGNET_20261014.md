# QAD matched48 recovery dragnet (Pop, 2026-10-14)

## Observed result

The protected 48-case packet was run with the same full prompt, temperature
0.6, one worker, per-case seeds, native executor and request limits against
clean QAD base and the direct-stage step-15000 weights. Each passed 1/48. The
base had 41 semantic mismatches, three invalid actions, two request-budget
exits and one incomplete result; no repetition stop. Step-15000 had 21
repetition stops, ten invalid actions, nine request-budget exits and seven
semantic mismatches. Both arms completed without infrastructure failures.
The source packet and trace artifacts are under
`/srv/storage/natlang-artifacts/qad-base-vs-15000-matched48-w1-20261014-v1`.
It is held evaluation material and must not enter training or repair queues.

The student now attempts useful tool work more often than the base, but its
own continuation frequently fails to recover from feedback. Among the 21
repetition endings, the final repeated action was `read_code` in 11, `eval` in
nine, and `return_result` in one. The last observation in seven carries a
specific parser/runtime correction: two invalid service-binding shadowings,
two undeclared fields, one TypeScript syntax error, one `ReferenceError`, and
one unbounded iteration. The other 14 include repeated help or service reads,
world-state feedback and an incomplete result. The final repeated action is
only a triage label, not necessarily the first mistake in a trajectory.

## Diagnostic work before an approach verdict

1. For each repeated `read_code`, compare consecutive request payloads,
   tool-result text, invocation scope and prompt history. Confirm that the
   model actually sees the returned source or help in the next call and that
   the result is not truncated or attached to a different child. Separate
   legitimate new-scope reads from no-progress same-scope reads.
2. For repeated `eval` failures, identify the first failed action, the exact
   one-sentence feedback, and whether the next action copied, mutated or
   corrected it. Check binding availability, executor rendering, error text,
   tokenization and any context-window loss. The repeated `const cancelTickets`
   declaration and `read_code(nl)` pattern are examples to trace, not prompts
   to teach the model.
3. Compare the step-15000 and clean-base distributions on the same early
   decisions. Check reasoning-to-action transition, stopping, context length,
   action decoding, loss masks and train/full-prompt proportions. The exact
   train-only action audit in `QAD_ACTION_TARGET_DRAGNET_20261014.md` found
   correct action suffixes fully supervised and prompt-token parity for two
   recovery decisions, but two decisions do not cover this packet.
4. Collect teacher recoveries from **disjoint train sources** for the common
   failure families, using real failing action and feedback handoffs. Admit
   only exact-oracle successes with correct continuation and held-group
   exclusion. Include them in the next declared broad union; then remeasure
   matched task behavior, not merely teacher-forced action NLL.

The present evidence does not distinguish insufficient recovery training from
a prompting, serving, state or template bug. Preserve that uncertainty while
running the checks above. The new Luna closed backlog contributes 23,651
approved train targets, including many nested semantic actions, but its render
and broad-union inclusion are still in progress; it was not part of step-15000.
