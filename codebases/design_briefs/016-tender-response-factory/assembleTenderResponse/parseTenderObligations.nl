---
description: Extract submission duties across tender body, schedules, forms and ordered addenda.
args:{goal:GoalContract;policy:PolicySelection;responses:TenderResponse[];evidence:EvidencePage[];addenda:EvidencePage[]}
returns:TenderRequirement[]
---
Enumerate each eligibility condition, scored criterion, conditional clause, response-form slot, naming rule and attachment requirement. Bind exact source edition and response location. Changed addendum clauses replace only explicitly superseded obligations; preserve unaffected prior requirements. Record explicit nonresponse decisions for unsupported optional claims.
For each tender source response and its ordered addendum context, bind obligations = interpretOneTenderSource(goal, policy, response, evidence, addenda). Append the source's returned duties with exact response locations; do not let a later addendum silently replace an earlier requirement unless it explicitly supersedes it.
