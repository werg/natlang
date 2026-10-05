/** Decisions and replay ordering for bounded, verified local rewriting. Not an MH sampler. */
export function serialTurnDriver(driver) {
  let pending=Promise.resolve();
  return request=>{
    const result=pending.then(()=>driver(request));
    pending=result.catch(()=>{});
    return result;
  };
}
export function trajectoryNll(turns) {
  turns=turns.filter(t=>t.training_target!==false);
  const tokens=turns.reduce((n,t)=>n+t.score.token_count,0);
  if(!tokens)throw Error('empty likelihood support');
  return -turns.reduce((n,t)=>n+t.score.sum_logprob,0)/tokens;
}
export function firstDifficultChunk(turns,controls) {
  for(const [turnIndex,turn] of turns.entries())
    if(turn.training_target!==false)for(const chunk of turn.score.chunks)
      if(chunk.mean_nll>controls.chunk_nll || chunk.max_nll>controls.token_nll)return {turnIndex,chunk};
  return null;
}
export function replaceChunk(response,chunk,value) {
  const out=structuredClone(response);
  if(chunk.kind==='action') {
    if(!value||typeof value!=='object')throw Error('action replacement must be structured');
    if(Array.isArray(value.calls))return structuredClone(value);
    if(!Array.isArray(value.tool_calls)&&typeof value.content!=='string')throw Error('action replacement missing content/calls');
    return {text:value.content??'',calls:(value.tool_calls??[]).map(c=>[c.function.name,typeof c.function.arguments==='string'?JSON.parse(c.function.arguments):c.function.arguments])};
  }
  if(chunk.kind==='sentence') {
    if(typeof value!=='string')throw Error('sentence replacement must be text');
    out.text=out.text.slice(0,chunk.value_start)+value+out.text.slice(chunk.value_end);return out;
  }
  if(!out.calls?.[chunk.call_index])throw Error('chunk call missing');
  if(chunk.kind==='tool_name') {
    if(typeof value!=='string'||!/^[$A-Z_a-z][$\w]*$/.test(value))throw Error('invalid tool name');
    out.calls[chunk.call_index][0]=value;
  } else if(chunk.kind==='code_line') {
    if(typeof value!=='string')throw Error('code replacement must be text');
    const original=out.calls[chunk.call_index][1][chunk.argument_name];
    out.calls[chunk.call_index][1][chunk.argument_name]=original.slice(0,chunk.value_start)+value+original.slice(chunk.value_end);
  } else if(chunk.kind==='argument'){
    const original=out.calls[chunk.call_index][1][chunk.argument_name];
    const kind=v=>v===null?'null':Array.isArray(v)?'array':typeof v;
    if(kind(value)!==kind(original))throw Error('replacement argument changed type');
    out.calls[chunk.call_index][1][chunk.argument_name]=value;
  }
  else throw Error('unsupported chunk kind');
  // Provider payload/usage belongs to the old action; do not carry it onto a rewrite.
  delete out.raw_response;delete out.raw_calls;delete out.reasoning;return out;
}
export function candidateDecision(candidate,current,controls) {
  if(!candidate.admitted)return {accepted:false,reason:'task_or_replay_rejected'};
  if(!Number.isFinite(candidate.ranking_nll))return {accepted:false,reason:'proposal_score_missing'};
  if(!Number.isFinite(candidate.chunk_max_nll)||candidate.chunk_nll>controls.chunk_nll||candidate.chunk_max_nll>controls.token_nll)return {accepted:false,reason:'student_chunk_too_difficult'};
  if(trajectoryNll(candidate.turns)>trajectoryNll(current.turns)+controls.regression_tolerance)
    return {accepted:false,reason:'trajectory_likelihood_regressed'};
  return {accepted:true,reason:'validated_teachable_rewrite'};
}
export function correctivePrefix(turns,turnIndex) {
  // Include one complete correct action; never create a mid-action EOS.
  return turns.filter(t=>t.training_admission?.approved && (t.decision?.index??t.decision_index)<=turnIndex);
}
