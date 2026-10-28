/** A source-declared, read-only delivery certificate; never infer it from an answer string. */
export type RequiredServiceReceipt = { service: 'world'; method: 'receipt' };

export async function checkRequiredServiceReceipt(
  spec: RequiredServiceReceipt | undefined, services: Record<string, unknown>, expected: unknown,
  same: (actual: unknown, expected: unknown) => boolean,
): Promise<boolean> {
  if (spec === undefined) return true;
  if (!spec || spec.service !== 'world' || spec.method !== 'receipt' || Object.keys(spec).length !== 2)
    throw new TypeError('required service receipt must name the exact read-only world receipt');
  const world = services.world as { receipt?: () => unknown } | undefined;
  if (!world || typeof world.receipt !== 'function') return false;
  try { return same(await world.receipt(), expected); }
  catch { return false; }
}
