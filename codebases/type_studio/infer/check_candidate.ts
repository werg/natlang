import { types } from 'natlang:services';
export default function check_candidate(target: Target, context: Context, candidate: Candidate): FitReport {
return types.check(target, context, candidate);
}
