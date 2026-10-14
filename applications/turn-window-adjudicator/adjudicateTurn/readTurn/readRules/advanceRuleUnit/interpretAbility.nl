---
description: Interpret one actual ability unit using already classified governing contexts.
args:
  unit: AbilityUnit
  unitClauses: SourceClause[]
  governingContexts: RuleContext[]
returns: AbilityInterpretation
---
1. Bind this actual unit's focus and original primary clauses. Read governingContexts as actual typed
   timing, identity and capacity judgments with their original witnesses. Locate the source identity and
   condition/effect roles of the proposed ability; unavailable decisive roles remain unresolved.
2. Declare local AbilityInterpretation and RuleMeaning with their supplied shapes. Create a scoped NL
   literal capturing governingContexts and unit.focus, and directly await it on these unitClauses.
   The literal returns meaning=null with a precise unresolved reason when no supported ability meaning
   can be formed; this typed unresolved contribution preserves earlier completed unit results.
3. Its numbered policy identifies whether the ability replaces an event before resolution, observes a
   resolved event as a trigger, or supplies a standing effect. Separate announced quantity, rewritten
   quantity and actual resolved effect as possible condition inputs. Context meaning governs this choice.
4. It then interprets controllerBasis as captured, current, last-known, none or unresolved from the
   source reference. A changed controller does not select a reference basis by itself. The condition
   and consequence may use different event versions; retain their actual source relation in timing.
5. Finally it identifies exact source token IDs, their stated capacities, effectClass and nonempty
   clause-keyed support. No token requirement means tokenIds empty when source establishes that absence.
   Missing source identity or decisive effect remains meaning=null; incomplete optional roles stay
   explicit in a supported RuleMeaning.unresolved instead of fabricated semantic defaults.
6. Return the actual AbilityInterpretation unchanged. Parent provenance checks bind original identity
   before consumption; this child decides ability roles, not the board result or window queue.
