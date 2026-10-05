/** Match old reference decisions to fresh visible inputs across reviewed harness changes.
 * System instructions and tool descriptions belong to the current harness. Keep task
 * instructions, argument observations and all non-bootstrap actions exact. Final
 * native replay and source admission remain mandatory; a match alone admits nothing.
 */
export function referenceObservationKey(messages) {
 const normalized=messages.filter(m=>m.role!=='system').map(m=>{
  const result={role:m.role,content:m.content??null};
  if(m.role==='user'&&typeof result.content==='string')result.content=result.content.replace(/\n\nIn eval you can use [^\n]*; the first eval below declares them\.\n\nEval also has the built-ins nl, iterateOn and transcript; read_code shows how to use each\.$/,'');
  if(m.tool_calls?.length)result.tool_calls=m.tool_calls.map(call=>({name:call.function.name,arguments:call.id==='scope_0'?'[host argument bootstrap]':call.function.arguments}));
  return result;
 });
 return JSON.stringify(normalized);
}
export function selectReferenceTurn(reference,used,request,key=referenceObservationKey) {
 const wanted=key(request.messages);
 return reference.find(turn=>!used.has(turn)&&key(turn.request.messages)===wanted);
}
