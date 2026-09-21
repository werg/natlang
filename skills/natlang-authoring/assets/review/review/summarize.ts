/*---
description: Retain assessments and count each verdict exactly.
args:
  assessments: Assessment[]
returns: Report
---*/
return {
  assessments: args.assessments,
  supported: args.assessments.filter(a => a.verdict === "supported").length,
  contradicted: args.assessments.filter(a => a.verdict === "contradicted").length,
  uncertain: args.assessments.filter(a => a.verdict === "uncertain").length,
};
