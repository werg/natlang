---
args:
  summary: string
returns: PromotionAnswer
---
You decide whether a case or a tier should move up, stay, or move down. summary is JSON about one subject:

- subject: "case" (one guarded piece of crisp code written for a function) or "tier" (one way of running a function, for example the student model or the specialized instructions).
- state: "shadow" (it is being checked and does not serve yet) or "active" (it serves calls).
- evidence, counted since the subject was created, or since a tier was last demoted:
  - served: calls it answered.
  - handed_off: calls it started and then could not finish correctly (wrong answer, failed check, crash in its own code). These count against it.
  - guard_misses: calls whose input it declined at its guard. These say nothing about quality. Ignore them.
  - infrastructure: attempts lost to a timeout, an unreachable model or a similar outage. This is bad luck, not a fault of the subject. Ignore them, unless they are all there is.
  - compared, worse, better: comparisons of its output with the reference executor's output on the same call. worse means it was judged worse than the reference. Replays (compared minus live_compared) were made on the calls the subject was written from, so they are weak evidence.
  - live_compared, live_worse: the comparisons on calls the subject had not seen. They are the strong evidence.
  - audited, audit_worse: later audits of calls the subject served.
  - recent_compared, recent_worse: the latest comparisons, up to 10.
- rule: the owner's limits. bound is the largest share of worse results allowed. comparisons and liveComparisons are the numbers of comparisons and of live comparisons needed before a promotion. auditMinimum and callMinimum are the numbers of audits and of served plus handed-off calls needed before the audits or hand-offs can demote.

Work in these steps and write down the numbers you use:

1. Failures. Count handed_off, worse and audit_worse as failures. guard_misses and infrastructure are outside the failure count.
2. Sample size. A share of 1 worse in 3 is a small sample. Promote when compared reaches comparisons and live_compared reaches liveComparisons. Demote when audited reaches auditMinimum, or served plus handed_off reaches callMinimum, or compared reaches comparisons; a demotion is also open when every one of the recent comparisons is worse (at least 5 of them).
3. Share. Compute worse divided by compared, live_worse divided by live_compared, audit_worse divided by audited, and handed_off divided by served plus handed_off. A share over rule.bound is bad.
4. Recency. If recent_compared is at least 5 and recent_worse is more than rule.bound times recent_compared, the subject is getting worse now, even if its whole history is fine. For an active subject that is a reason to demote when step 3 is also close to the bound. For a shadow subject it is a reason to wait.
5. Decide.
   - state shadow: answer "promote" when the sample sizes of step 2 are met, no share in step 3 is bad, and step 4 shows no recent trend. Otherwise "keep".
   - state active: answer "demote" when a share in step 3 is bad on a large enough sample, or when step 4 applies. Otherwise "keep".
   - In every other case answer "keep".

When the numbers are close to a limit, keep: moving a subject in the wrong direction costs more than waiting for more evidence.

Answer with decision ("promote", "keep" or "demote") and a reason of one or two sentences that names the numbers you used.
