/** Authored plans describe intended actions only; no hidden gold, invented observation or justification. */
export const ACTION_PLAN_VERSION = 'authored-action-plans/1';
export function actionPlan(calls) {
  return calls.map(([tool, args]) => {
    if (tool === 'eval') {
      const code = String(args.code ?? '');
      if (/\biterateOn\s*\(/.test(code)) return 'I will run the step with iterateOn, check its stopping condition, and obtain the requested result from the final state.';
      if (/\bnl(?:\.with\([^)]*\))?(?:<[^`]*?>)?`/.test(code)) return 'I will pass each relevant item and criterion to an inline natural-language function, await its typed result, and combine the results with exact code.';
      return 'I will execute the next computation or operation in eval and inspect its result.';
    }
    if (['read_file','read_page','read_code','list_files','search_files'].includes(tool)) return 'I will inspect the requested source before deciding the next step.';
    if (['write_file','edit_file','edit_code'].includes(tool)) return 'I will apply the specified edit and inspect the resulting state.';
    if (tool === 'return_result') return args.status === 'success' ? 'I will return the answer in its declared type.' : 'I will report the stopping status and the reason this call cannot continue.';
    return 'I will perform the next available operation and inspect its result.';
  }).join(' ');
}
