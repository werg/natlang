import type { ExactQuantityCheck, QuantityLimitCheck } from "../../types.js";

type Parsed = { integer: bigint; scale: number };
function parse(value: string): Parsed {
  if (!/^(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$/.test(value)) throw new Error("invalid_nonnegative_decimal");
  const [whole, fraction = ""] = value.split(".");
  return { integer: BigInt(whole + fraction), scale: fraction.length };
}
function render(integer: bigint, scale: number): string {
  if (scale === 0) return integer.toString();
  const padded = integer.toString().padStart(scale + 1, "0");
  const fraction = padded.slice(-scale).replace(/0+$/, "");
  return fraction ? `${padded.slice(0, -scale)}.${fraction}` : padded.slice(0, -scale);
}
export default function verifyQuantityLimits(checks: QuantityLimitCheck[]): ExactQuantityCheck[] {
  return checks.map((check) => {
    const issues: string[] = [];
    let available: Parsed;
    try { available = parse(check.available.value); }
    catch { available = { integer: 0n, scale: 0 }; issues.push("invalid_available_quantity"); }
    const seen = new Set<string>(), values: Parsed[] = [];
    for (const item of check.assignments) {
      if (seen.has(item.id)) issues.push(`duplicate_assignment:${item.id}`);
      seen.add(item.id);
      if (item.quantity.unit !== check.available.unit) issues.push(`unit_mismatch:${item.id}`);
      try { values.push(parse(item.quantity.value)); }
      catch { issues.push(`invalid_quantity:${item.id}`); }
    }
    const scale = Math.max(available.scale, ...values.map((item) => item.scale), 0);
    const total = values.reduce((sum, item) => sum + item.integer * 10n ** BigInt(scale - item.scale), 0n);
    const limit = available.integer * 10n ** BigInt(scale - available.scale);
    if (total > limit) issues.push("quantity_exceeds_limit");
    if (check.evidenceRefs.length === 0) issues.push("limit_without_evidence");
    return {
      limitId: check.limitId, limitKind: check.limitKind,
      total: { value: render(total, scale), unit: check.available.unit }, available: check.available,
      valid: issues.length === 0, issue: issues.join(";"), evidenceRefs: [...check.evidenceRefs],
    };
  });
}
