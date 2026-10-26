---
args: { history: ObservedAction[], transition: PracticeWorldTransition, challenge: SeededPracticeChallenge }
returns: ProcedureDebrief
---
For each procedure obligation, locate the actual learner action that satisfies it, violates it, or leaves it unobserved. Compare the recorded sequence with the required checks and escalation steps. Assess the final outcome separately: a fortunate result cannot erase a missed check, and an adverse simulated result does not make a correct procedure unsound. Cite each feedback statement to action IDs.
