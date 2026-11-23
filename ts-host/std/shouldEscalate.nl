---
description: Decide whether a local decision should escalate to a stronger fallback model.
args:
  decision: DecisionSummary
  floor?: number
  risk?: Record<string, number>
  history?: EscalationHistory
  cost?: TierCost
returns: boolean
readout: decision
types:
  DecisionSummary: "{ value: unknown, confidence: number, scored: boolean, probabilities: { value: unknown, probability: number }[], fields?: Record<string, { value: unknown, probability: number }[]> }"
  EscalationHistory: "{ totalCalls?: number, escalationRate?: number, disagreementRate?: number }"
  TierCost: "{ local?: number, fallback?: number, ratio?: number }"
---
decision is the local model's answer and probability distribution. floor is the minimum confidence expected for a local
answer; risk maps high-stakes labels to the probability above which escalation is warranted even when another label won
argmax. history (when present) summarizes this function's earlier runs: how many there were, how often they escalated,
and how often the fallback answered differently from the local model; cost (when present) compares the local and
fallback tiers, in milliseconds per call when measured from those runs.

Answer true when the decision should escalate to the fallback model: confidence is below floor, a high-risk label has
material probability relative to its risk threshold, the distribution is ambiguous on a high-stakes case, or prior
history shows frequent fallback disagreements under similar uncertainty. Answer false when the local decision is
well-supported and safe to serve locally.
