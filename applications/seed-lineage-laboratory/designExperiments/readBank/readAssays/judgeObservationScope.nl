---
description: Decide what one observation can establish under its actual protocol.
args:
  observationClauses: SourceClause[]
  protocolClauses: SourceClause[]
returns: ScopeDecision
---
1. Bind only the supplied original observationClauses and protocolClauses. ScopeMeaning is the
   declared semantic result with measuresClaim, quantity, compartment, stage, regime, reason and
   evidence:{clauseId:string;exactQuote:string}[]. Its result is one observation-scope judgment.
2. Use this positive direct-await form; the literal receives actual observation clauses and captures
   the actual protocol clauses. Define the local alias and bind the returned meaning in the same action:

   type ScopeMeaning = {measuresClaim:boolean|null;quantity:string;compartment:string;stage:string;regime:string;reason:string;evidence:{clauseId:string;exactQuote:string}[]};
   const judgeScope = nl.with<ScopeMeaning>({protocolClauses})`1. Identify the observation's reported quantity and tissue owner. 2. Resolve stage, regime and readout applicability from the supplied protocol clauses, keeping missing calibration or ambiguous ownership unresolved. 3. Return one ScopeMeaning directly with true for established applicable measurement, false for established mismatch, null for missing decisive scope, and nonempty clause-keyed exact supporting quotes.`;
   const meaning = await judgeScope(observationClauses);

3. Consume that actual meaning. Match every evidence.clauseId to a supplied original observation or
   protocol clause; require each exactQuote to be nonempty and contained in its original clause witness.
   Bind sourceId/path from the matched witness. Invalid provenance identifies the original clause for repair.
4. Return ScopeDecision by copying the actual semantic fields and the checked original witnesses.
   A supported unknown scope retains measuresClaim=null and its precise missing premise. A failed call
   is a failed contribution requiring repair, rather than an independently guessed observation judgment.
5. This local contract identifies quantity, compartment and applicability. Later predictObservedTrait
   derives biological answers and assessObservation derives likelihoods from this actual scope. A valid
   coat reading remains evidence about coat even when the requested mechanism concerns adult expression.
