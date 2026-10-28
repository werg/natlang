/** A source-declared, read-only delivery certificate; never infer it from an answer string. */
export type RequiredServiceReceipt = { service: 'world'; method: 'receipt' } |
  { service: 'greenhouse'; method: 'certificate' };

export async function checkRequiredServiceReceipt(
  spec: RequiredServiceReceipt | undefined, services: Record<string, unknown>, expected: unknown,
  same: (actual: unknown, expected: unknown) => boolean,
): Promise<boolean> {
  if (spec === undefined) return true;
  if (!spec || Object.keys(spec).length !== 2 || !(
    (spec.service === 'world' && spec.method === 'receipt') ||
    (spec.service === 'greenhouse' && spec.method === 'certificate')))
    throw new TypeError('required service receipt must name a supported read-only certificate');
  const service = services[spec.service] as Record<string, (() => unknown) | undefined> | undefined;
  const method = service?.[spec.method];
  if (typeof method !== 'function') return false;
  try { return same(await method.call(service), expected); }
  catch { return false; }
}
