---
description: Try one finite deletion pass using actual scoped sufficiency judgments over the affected proof cone.
args:
  state: SupportPassState
returns: SupportPassState
---
1. If state.settled return state. Copy candidateIds into passIds in stable source order, and keep a
   separate currentCandidates list that later judgments will observe after earlier accepted deletions.
2. Capture the changed obligation key, before/after consequences, existing background proof and
   source authority/availability rules. In eval declare SupportJudgment with sufficient:boolean|null,
   requiredPremiseIds:string[], reason:string and witnesses:Witness[]. Create nl.with<SupportJudgment>.
3. For each candidate still in currentCandidates, form proposedCandidates by removing that one ID.
   Gather the actual affected proof cone and exact source premises available under the proposal.
4. Directly await the literal on that changing proposed premise set. Ask whether the same after-status,
   amount alternatives and permission result are entailed from before plus those premises. Reevaluate
   evidence availability and open conditions; a removed receipt assertion cannot survive as a cached date.
5. Consume sufficient=true by replacing currentCandidates with proposedCandidates and appending
   the removed ID. Consume false by retaining the candidate and its necessity reason. Consume null by
   retaining the candidate and a typed unresolved reason; uncertainty never licenses deletion.
6. Validate that requiredPremiseIds names only retained candidates or declared background premises.
   A judgment using an omitted premise is unresolved and cannot change currentCandidates.
7. After all passIds have been tried, return updated candidateIds, removedIds, reasons and witnesses.
   If no deletion occurred set settled=true. Otherwise leave settled=false for another strictly smaller pass.
8. The parent's next pass consumes this returned list. The literal's boolean therefore changes the
   typed search state and the final explanation; a standalone prose explanation does not replace it.
