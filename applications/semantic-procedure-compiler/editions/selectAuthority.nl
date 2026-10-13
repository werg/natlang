---
description: Judge actual adoption and issuer competence under the source-defined procedure authority.
args:
  records: SourceRecord[]
  context: SourceRecord[]
returns: AuthorityDecision[]
---
1. Inventory the visible authority grant and claimed procedure editions. Separate who may issue a
   safety procedure, who may waive a queue rule and who merely records how past operators behaved.
2. For each edition capture its original record, governing authority clauses and claimed effective
   scope. In eval declare AuthorityJudgment with adopted:boolean|null, competent:boolean|null,
   scope:string[], reason:string and quotes:string[]. Construct a scoped NL literal over that context.
3. Directly await the literal on the edition's claim. Its instructions are: judge the source-supported
   act of adoption separately from issuer power; narrow competence to the granted subject; preserve
   an absent adoption or grant premise as unknown instead of treating publication as authorization.
4. Consume adopted/competent/scope into AuthorityDecision. Bind returned quote witnesses to the
   original record/context identities after exact quotation matching; unsupported quotations are
   unresolved evidence and cannot certify the edition's claimed scope.
5. Record LocalDecision only after this returned judgment, naming the edition ID and actual scope
   findings. Later supersession uses these fields to select or reject source clauses.
6. Return one decision per inventoried edition. A queue waiver authority cannot waive safety duties,
   and a named actor's favorite old policy cannot become a current adopted edition by repetition.
