export default function check_calls(context: Context, claims: CallClaim[]): Diagnostic[] {
return fx.types.calls(context, claims);
}
