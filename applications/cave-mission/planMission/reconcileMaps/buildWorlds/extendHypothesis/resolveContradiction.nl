---
description: Decide whether overlapping survey claims genuinely conflict under one proposed transform.
args:
  claims: PassageAssertion[]
  hypothesis: FrameHypothesis
  survey: SurveyEvidence
returns: AnchorAssessment
---
1. Condition the entire judgment on the supplied hypothesis: its transform and bindings define this
   candidate physical world. Read the original frame ownership of each claim, then translate its
   endpoints through those bindings. The actual-world choice remains unresolved across hypotheses.
2. Check whether any claim still lacks a required endpoint binding or source scope even within this
   hypothesis. If so, retain the precise missing premise for compatible:null with exact witnesses.
3. Compare claims about the same interpreted physical passage. Distinguish existence, direction,
   carried-state permission and cost. A present narrow tunnel and a rule blocking a full carrier can
   both hold; an explicit absent passage and present passage in the same scope cannot.
4. Interpret one-way statements in their source direction. Read temporal scope before combining
   apparently opposed movement claims. The current-world boundary governs when the evidence supplies
   one; scope differences justify compatibility only when supported by the actual source clauses.
5. Return compatible:false only for a witnessed contradiction that persists under this supplied
   hypothesis, with the exact conflicting premises. Uncertainty over which complete hypothesis is
   actual is separate from this conditional consistency judgment.
6. Return compatible:true when all translated claims can hold together in this candidate world and
   no required conditional premise is missing. Include the source witnesses that establish the shared
   endpoints and the compatible directional, existence and carried-state scopes.
7. Return compatible:null only for the specific missing conditional premise identified above. Keep
   all quotes exact. The caller consumes true into this world's topology, false into rejection, and
   null into unresolved evidence requiring repair before successful planning.
