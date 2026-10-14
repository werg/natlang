---
description: Decide what one observation can establish under its actual protocol.
args:
  observationClauses: SourceClause[]
  protocolClauses: SourceClause[]
returns: ScopeDecision
---
1. Bind the actual observation and protocol records. Declare a local ScopeMeaning containing the
   ScopeDecision semantic fields and clause-keyed quote support, then directly await a scoped literal.
2. The literal first identifies the reported biological quantity and tissue owner. It next resolves
   developmental stage and regime against the protocol. Finally it decides whether the observation
   measures the claimed expression quantity at the supplied resolution.
3. Return true for established applicable measurement, false for an explicit compartment/regime
   mismatch, and null for missing calibration or ambiguous tissue ownership. Preserve the reason.
4. Consume the literal's actual result. For every supporting quote, bind original clause identity and
   require nonempty exact text. Missing support retains a null scope rather than a confident measurement.
5. Return the typed ScopeDecision. Its quantity and compartment remain visible to later prediction;
   a maternal coat reading can be valid evidence about coat while irrelevant to adult expression.
