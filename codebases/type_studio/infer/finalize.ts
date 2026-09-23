export default function finalize(candidate: Candidate, fit: FitReport): Assessment {
return { candidate: candidate, fit: fit,
         status: !fit.parseable || !fit.obligations_ok ? "invalid" :
                 fit.checked > 0 && candidate.alternatives.length === 0 ? "consistent" : "uncertain" };
}
