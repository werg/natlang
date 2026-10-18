# Captured answer equivalence continuation

The semantic decision remains the source's natural-language span rubric and an
independent natural-language review of the exact captured answer. This is the
existing `OracleJudge` boundary, supplied from a sealed review rather than a new
model invocation. Crisp code authenticates evidence and transports that verdict.
No substring rule, new answer, regenerated trajectory, or exact-string verdict is
introduced. Ordinary span comparisons and live model prompts remain unchanged.

1. Authenticate the immutable review, its inventory, the original raw output,
   complete original program, question, reference, rubric, and captured answer.
   Require an independent approval and a nonempty reason for this exact TRAIN
   row. Protected splits, changed rows, missing/contradictory approvals and any
   continuation of an already derived row fail closed.
2. Reproduce the original current span comparison without a judge. Require the
   captured `needs_review` result, a completed execution, only the answer check
   failing, and every declared nonanswer check passing. Preserve full original
   files, effects, trajectory, inputs, and original negative evaluation as custody.
3. Call the same current span oracle with its existing judge interface. The
   judge may return only the authenticated independent verdict for the exact
   actual/reference/context/rubric tuple. Keep the oracle level `span` and its
   lexical score; record the explicit semantic reason and authority reference.
4. Attach the derived evaluation through the existing teacher quality
   continuation. Final verification reruns steps 1–3 and compares the complete
   derived row to the authenticated original, preserving all executable capture
   and target bytes. Existing split/source/decision-hold checks remain mandatory.
5. Test authentic current span execution, four real captured approvals, changed
   answer/program/question/reference/rubric/receipt/raw bytes, wrong review,
   unrelated exact failures, failed files/effects, and protected/non-TRAIN rows.
   No parent is admitted until independent review and the normal final pipeline.

The review performs semantic policy. Python verifies binding and nonanswer
eligibility; the unchanged TypeScript oracle evaluates comparison and invokes the
bound verdict. Durable receipts and the existing converter supply publication
and original-row custody. This offline authority plumbing adds no live prompt or
feedback text and carries no additional live-sampling obligation.
