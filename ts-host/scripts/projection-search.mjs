/** Probability accounting for request-boundary suffix projection of interactive episodes. */
export function mhDecision(current, candidate, cut, draw) {
  if (!(draw > 0 && draw < 1) || !Number.isInteger(cut) || cut < 0) throw new Error('invalid MH draw/cut');
  const sum = (turns, key) => turns.reduce((v,t) => v+t[key],0);
  for (const state of [current,candidate]) {
    if (!state.length || state.some(t => !Number.isFinite(t.target) || !Number.isFinite(t.proposal)))
      throw new Error('missing finite target/proposal likelihood');
  }
  if (cut >= current.length || cut >= candidate.length) throw new Error('cut outside shared prefix');
  // The cut is sampled from a fixed, state-independent set. Identical prefix
  // probabilities cancel. q(reverse) is evaluated on the original suffix with
  // its ORIGINAL, freshly replayed observation contexts, not candidate contexts.
  const targetDelta = sum(candidate,'target')-sum(current,'target');
  const forward = sum(candidate.slice(cut),'proposal');
  const reverse = sum(current.slice(cut),'proposal');
  const logAcceptance = Math.min(0,targetDelta+reverse-forward);
  return {accepted:Math.log(draw)<logAcceptance, log_acceptance:logAcceptance,
    target_delta:targetDelta, forward_logprob:forward, reverse_logprob:reverse, draw};
}
export function canonical(value) {
  if(Array.isArray(value)) return '['+value.map(canonical).join(',')+']';
  if(value && typeof value==='object') return '{'+Object.keys(value).sort().map(k=>JSON.stringify(k)+':'+canonical(value[k])).join(',')+'}';
  return JSON.stringify(value);
}
export function randomStream(seed) {
  let s=BigInt(seed)&((1n<<64n)-1n);
  return () => {s=(s+0x9e3779b97f4a7c15n)&((1n<<64n)-1n);let z=s;
    z=((z^(z>>30n))*0xbf58476d1ce4e5b9n)&((1n<<64n)-1n);
    z=((z^(z>>27n))*0x94d049bb133111ebn)&((1n<<64n)-1n);z^=z>>31n;
    return (Number(z>>12n)+0.5)/4503599627370496;};
}
