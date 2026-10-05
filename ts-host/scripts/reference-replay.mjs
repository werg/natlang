/** Match old reference decisions to fresh visible inputs across reviewed harness changes.
 * System instructions and tool descriptions belong to the current harness. Keep task
 * instructions, argument observations and all non-bootstrap actions exact. Final
 * native replay and source admission remain mandatory; a match alone admits nothing.
 */
function redundantLocalSummary(content,previous) {
 if(typeof content!=='string'||!content.startsWith('console:\n')||previous?.role!=='assistant')return content;
 const match=content.match(/\nStored local ([A-Za-z_$][\w$]*) = [\s\S]*$/);
 if(!match)return content;
 const prefix=content.slice(0,match.index);
 if(!prefix.endsWith('\nnull'))return content;
 const printed=prefix.slice('console:\n'.length,-'\nnull'.length);
 try{const value=JSON.parse(printed);if(value===null||typeof value!=='object')return content;}catch{return content;}
 const declaration=match[1].replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
 const direct=new RegExp('console\\.log\\(\\s*JSON\\.stringify\\(\\s*'+declaration+'\\s*[,)]');
 const fullValuePrinted=previous.tool_calls?.some(call=>{
  if(call.function?.name!=='eval')return false;
  try{const args=typeof call.function.arguments==='string'?JSON.parse(call.function.arguments):call.function.arguments;return typeof args?.code==='string'&&direct.test(args.code);}catch{return false;}
 });
 return fullValuePrinted?prefix:content;
}

export function referenceObservationKey(messages) {
 const normalized=messages.filter(m=>m.role!=='system').map((m,index,visible)=>{
  const result={role:m.role,content:m.role==='tool'?redundantLocalSummary(m.content,visible[index-1]):m.content??null};
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
