/** Optional source-declared service call obligations, observed on the real service object. */
export type RequiredServiceCall = {
  service: string; method: string; args: unknown[]; min_successes: number; max_successes: number | null;
};
type Observation = { service: string; method: string; args: unknown[] | undefined; success: boolean };

function snapshot(value: unknown): unknown[] | undefined {
  try { return JSON.parse(JSON.stringify(value)) as unknown[]; } catch { return undefined; }
}

export function observeRequiredServiceCalls(specs: RequiredServiceCall[] | undefined, services: Record<string, object>) {
  if (specs === undefined) return () => true;
  if (!Array.isArray(specs) || !specs.length) throw new TypeError('required service calls must be a nonempty list');
  const observations: Observation[] = [];
  const seen = new Set<string>();
  for (const spec of specs) {
    if (!spec || Object.keys(spec).sort().join(',') !== 'args,max_successes,method,min_successes,service' ||
        typeof spec.service !== 'string' || typeof spec.method !== 'string' ||
        !/^[A-Za-z_][A-Za-z0-9_]*$/.test(spec.service) || !/^[A-Za-z_][A-Za-z0-9_]*$/.test(spec.method) ||
        !Array.isArray(spec.args) || !Number.isInteger(spec.min_successes) ||
        (spec.max_successes !== null && !Number.isInteger(spec.max_successes)) ||
        spec.min_successes < 1 || (spec.max_successes !== null && spec.max_successes < spec.min_successes))
      throw new TypeError('required service call has an invalid exact contract');
    const key = `${spec.service}.${spec.method}`;
    if (seen.has(key)) throw new TypeError('required service call methods must be unique');
    seen.add(key);
    const original = services[spec.service] as Record<string, unknown> | undefined;
    const method = original?.[spec.method];
    if (!original || typeof method !== 'function') continue;
    const wrapped = {...original};
    wrapped[spec.method] = (...args: unknown[]) => {
      const observation: Observation = {service: spec.service, method: spec.method, args: snapshot(args), success: false};
      observations.push(observation);
      const result = Reflect.apply(method, original, args);
      if (result && typeof (result as PromiseLike<unknown>).then === 'function')
        return Promise.resolve(result).then(value => { observation.success = true; return value; });
      observation.success = true;
      return result;
    };
    services[spec.service] = wrapped;
  }
  return (same: (actual: unknown, expected: unknown) => boolean) => specs.every(spec => {
    const successful = observations.filter(x => x.service === spec.service && x.method === spec.method && x.success);
    const matches = successful.filter(x => x.args !== undefined && same(x.args, spec.args));
    return matches.length >= spec.min_successes && (spec.max_successes === null || successful.length <= spec.max_successes);
  });
}
