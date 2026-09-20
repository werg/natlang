/*---
args:
  candidate: Candidate
  fit: FitReport
returns: Assessment
---*/
return { candidate: args.candidate, fit: args.fit,
         status: !args.fit.parseable || !args.fit.obligations_ok ? "invalid" :
                 args.fit.checked > 0 && args.candidate.alternatives.length === 0 ? "consistent" : "uncertain" };
