import { execFileSync } from 'node:child_process';
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
export function cloudflareLogin() {
  const suppliedToken=process.env.CLOUDFLARE_API_TOKEN ?? process.env.CLOUDFLARE_AUTH_TOKEN;
  if(process.env.CLOUDFLARE_ACCOUNT_ID && suppliedToken) {
    if(!/^[a-f0-9]{32}$/i.test(process.env.CLOUDFLARE_ACCOUNT_ID))throw new Error('Invalid CLOUDFLARE_ACCOUNT_ID');
    return {account:process.env.CLOUDFLARE_ACCOUNT_ID,token:suppliedToken};
  }
  const identity=JSON.parse(execFileSync('wrangler',['whoami','--json'],{encoding:'utf8',stdio:['ignore','pipe','pipe']}));
  const account=process.env.CLOUDFLARE_ACCOUNT_ID ?? (identity.accounts?.length===1 ? identity.accounts[0].id : undefined);
  if(!account || !/^[a-f0-9]{32}$/i.test(account)) throw new Error('Set CLOUDFLARE_ACCOUNT_ID to select a Wrangler account.');
  const token=process.env.CLOUDFLARE_API_TOKEN ?? process.env.CLOUDFLARE_AUTH_TOKEN ?? JSON.parse(execFileSync('wrangler',['auth','token','--json'],{encoding:'utf8',stdio:['ignore','pipe','pipe']})).token;
  if(!token) throw new Error('No Cloudflare token available. Run wrangler login.');
  return {account,token}; // Private in-memory credentials; never persist or log this object.
}
export async function clefDecision(login,model,request) {
  if(!['clef','clef-flash'].includes(model))throw new Error('Unsupported decision model');
  const body={model,state:request.state,questions:request.questions};
  for(let attempt=0;attempt<4;attempt++) {
    let response;
    try {
      response=await fetch(`https://api.cloudflare.com/client/v4/accounts/${login.account}/ai/run/@cf/cloudflare/${model}`,{
        method:'POST',headers:{Authorization:`Bearer ${login.token}`,'Content-Type':'application/json'},body:JSON.stringify(body),signal:AbortSignal.timeout(120000)});
    } catch {
      if(attempt===3)throw new Error('Workers AI transport failure after four attempts');
      await sleep(Math.min(30000,2000*2**attempt)*(0.8+Math.random()*0.4));continue;
    }
    if(response.status===429 || response.status>=500) {
      if(attempt===3)throw new Error(`Workers AI HTTP ${response.status} after four attempts`);
      const retry=response.headers.get('retry-after');
      const seconds=retry&&Number.isFinite(Number(retry))?Number(retry):retry?Math.max(0,(Date.parse(retry)-Date.now())/1000):0;
      await response.body?.cancel();
      await sleep(Math.max(seconds*1000,Math.min(30000,2000*2**attempt)*(0.8+Math.random()*0.4)));continue;
    }
    const envelope=await response.json();
    if(!response.ok || envelope.success!==true)throw new Error(`Workers AI rejected request: HTTP ${response.status}; error codes ${(envelope.errors??[]).map(x=>x.code).join(',')}`);
    const result=envelope.result,answer=result?.answers?.decision;
    const allowed=Object.keys(request.questions.decision.criteria), probs=answer?.probabilities;
    if(answer?.type!=='choice' || !allowed.includes(answer.choice) || !probs || Object.keys(probs).length!==allowed.length ||
      !allowed.every(key=>Number.isFinite(probs[key])&&probs[key]>=0&&probs[key]<=1) || Math.abs(allowed.reduce((sum,key)=>sum+probs[key],0)-1)>0.01 ||
      !Number.isSafeInteger(result.usage?.input_tokens) || result.usage.input_tokens<0)throw new Error('Invalid typed decision or usage response');
    return result;
  }
}
