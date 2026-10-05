import {setTimeout as sleep} from 'node:timers/promises';

/** Retry inference transport only. Callers must not use this for state-changing tools. */
export async function postInferenceJson(url,body,{signal,fetchImpl=fetch,wait=sleep,onRetry=()=>{},retries=3,baseMs=5000}={}) {
 const payload=JSON.stringify(body);
 for(let attempt=0;;attempt++){
  signal?.throwIfAborted();
  let error;
  try {
   const response=await fetchImpl(url,{method:'POST',headers:{'content-type':'application/json'},body:payload,signal});
   if(!response.ok){
    error=Object.assign(new Error('provider HTTP '+response.status),{status:response.status,headers:response.headers});
    await response.body?.cancel();
    throw error;
   }
   return await response.json();
  }catch(caught){error=caught;}
  signal?.throwIfAborted();
  const code=error.cause?.code??error.code;
  const retryable=[408,429,500,502,503,504].includes(error.status)||
    ['UND_ERR_SOCKET','UND_ERR_CONNECT_TIMEOUT','UND_ERR_HEADERS_TIMEOUT','UND_ERR_BODY_TIMEOUT','ECONNRESET','ECONNREFUSED','EPIPE','ETIMEDOUT'].includes(code);
  if(!retryable||attempt>=retries)throw error;
  const retryAfter=error.headers?.get('retry-after');
  const seconds=retryAfter===null||retryAfter===undefined?NaN:Number(retryAfter);
  const requested=Number.isFinite(seconds)?seconds*1000:Date.parse(retryAfter)-Date.now();
  const delay=Math.max(Number.isFinite(requested)?requested:0,Math.min(30000,baseMs*2**attempt));
  await onRetry({attempt:attempt+1,wait_ms:delay,status:error.status??null,code:code??null});
  await wait(delay,undefined,{signal});
 }
}
