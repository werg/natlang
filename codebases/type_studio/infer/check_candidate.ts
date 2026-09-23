export default function check_candidate(target: Target, context: Context, candidate: Candidate): FitReport {
return fx.types.check(target, context, candidate);
}
