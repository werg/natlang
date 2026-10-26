# Misconception repair studio

`learningRepairSession.nl` is a learner-session state machine, not a fixed five-stage pipeline. It checks task clarity and answer-space validity before interpreting a response. Ambiguous prompts go directly to task repair and produce no learner diagnosis. For clear tasks, it records visible work separately from a conservative set of competing explanations, then designs a neutral one-feature probe only if those explanations make different predictions.

The diagnostic branch hides hypothesis labels. Authorized accommodations change presentation while a separate competency-invariance check preserves the target skill, prerequisite set, and accepted answers. Repair content passes a subject-matter verification gate and uses a bounded scaffold sequence. Transfer requires a novel surface and no conceptual hint; assisted success never becomes mastery evidence. The saved checkpoint records last completed task, private labels, accommodation, and hint exposure so an interrupted session resumes without revealing diagnostic intent. Persistent ambiguity or frustration creates one bounded instructor handoff.

Five exact gate procedures are relocated beneath `learningRepairSession/{response,hypotheses,diagnostic,repair,transfer}/audit/qualify.nl`. Their typed cases and expected verdicts are unchanged; only root paths move.
