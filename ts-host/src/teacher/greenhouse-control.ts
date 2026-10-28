/** Observe source-declared greenhouse control calls without altering the service. */
export type RequiredGreenhouseControl = { kind: 'iterate-on-temperature-control' };
type Call = { method: string; args: unknown[]; result: unknown; at: number; success: boolean };

export function observeGreenhouseControl(spec: RequiredGreenhouseControl | undefined, services: Record<string, object>,
  traceLength: () => number) {
  if (spec === undefined) return (_events: readonly Record<string, unknown>[]) => true;
  if (!spec || spec.kind !== 'iterate-on-temperature-control' || Object.keys(spec).length !== 1)
    throw new TypeError('unsupported greenhouse control contract');
  const source = services.greenhouse as Record<string, unknown> | undefined;
  const calls: Call[] = [];
  if (source) {
    const wrapped = {...source};
    for (const name of ['set', 'read', 'certificate']) {
      const method = source[name];
      if (typeof method !== 'function') continue;
      wrapped[name] = (...args: unknown[]) => {
        const call: Call = {method:name,args:structuredClone(args),result:undefined,at:traceLength(),success:false};
        calls.push(call);
        const value = Reflect.apply(method, source, args);
        if (value && typeof (value as PromiseLike<unknown>).then === 'function')
          return Promise.resolve(value).then(result => {call.result=structuredClone(result);call.success=true;return result});
        call.result=structuredClone(value);call.success=true;return value;
      };
    }
    services.greenhouse=wrapped;
  }
  return (events: readonly Record<string, unknown>[]) => {
    const marks=events.map((event,index)=>({event,index})).filter(x=>x.event.kind==='iteration_step');
    const ids=new Set(marks.map(x=>x.event.iteration_id));
    if(ids.size!==1 || !marks.some(x=>x.event.phase==='done')) return false;
    const initial=marks.find(x=>x.event.phase==='initial');
    const steps=marks.filter(x=>x.event.phase==='step');
    if(!initial||!steps.length) return false;
    const prior=calls.filter(x=>x.success&&x.method==='read'&&x.at<=initial.index).at(-1);
    let temp=(prior?.result as {temp?:unknown} | undefined)?.temp;
    if(typeof temp!=='number'||!Number.isFinite(temp)) return false;
    let boundary=initial.index;
    for(const step of steps){
      const part=calls.filter(x=>x.success&&x.at>boundary&&x.at<=step.index);
      if(part.length!==3 || part[0]?.method!=='set' || part[1]?.method!=='read' || part[2]?.method!=='certificate') return false;
      const controls=part[0].args[0] as {heater?:unknown;vent?:unknown} | undefined;
      if(!controls || Object.keys(controls).sort().join(',')!=='heater,vent' ||
          controls.heater!==(temp<20.5) || controls.vent!==(temp>21.5)) return false;
      temp=(part[1].result as {temp?:unknown} | undefined)?.temp;
      if(typeof temp!=='number'||!Number.isFinite(temp)) return false;
      boundary=step.index;
    }
    return true;
  };
}
