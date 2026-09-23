import { types } from 'natlang:services';
export default function check_calls(context: Context, claims: CallClaim[]): Diagnostic[] {
return types.calls(context, claims);
}
