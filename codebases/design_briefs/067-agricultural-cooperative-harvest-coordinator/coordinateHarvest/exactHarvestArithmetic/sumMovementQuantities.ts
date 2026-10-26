import type { MovementReceipt, MovementReceiptSum, Quantity } from "../../types.js";

type DecimalValue = { units: bigint; scale: number };
type Group = { unit: string; scale: number; units: bigint; invalid: boolean };

function parseQuantity(value: string): DecimalValue {
  if (!/^(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$/.test(value)) throw new Error("invalid_nonnegative_decimal");
  const [whole, fraction = ""] = value.split(".");
  return { units: BigInt(whole + fraction), scale: fraction.length };
}

function render(units: bigint, scale: number): string {
  if (scale === 0) return units.toString();
  const padded = units.toString().padStart(scale + 1, "0");
  const whole = padded.slice(0, -scale);
  const fraction = padded.slice(-scale).replace(/0+$/, "");
  return fraction.length === 0 ? whole : `${whole}.${fraction}`;
}

function fingerprint(receipt: MovementReceipt): string {
  return JSON.stringify([receipt.lotId, receipt.phase, receipt.quantity.value, receipt.quantity.unit,
    receipt.observedAt, receipt.sourceVersion]);
}

export default function sumMovementQuantities(receipts: MovementReceipt[]): MovementReceiptSum {
  const firstById = new Map<string, MovementReceipt>();
  const duplicateReceiptIds = new Set<string>();
  const invalidReceiptIds = new Set<string>();
  const conflictingIds = new Set<string>();

  for (const receipt of receipts) {
    const previous = firstById.get(receipt.id);
    if (!previous) firstById.set(receipt.id, receipt);
    else {
      duplicateReceiptIds.add(receipt.id);
      if (fingerprint(previous) !== fingerprint(receipt)) {
        conflictingIds.add(receipt.id);
        invalidReceiptIds.add(receipt.id);
      }
    }
  }

  const groups = new Map<string, Group>();
  for (const [id, receipt] of firstById) {
    const key = `${receipt.lotId}|${receipt.phase}`;
    if (conflictingIds.has(id)) {
      groups.set(key, { unit: receipt.quantity.unit, scale: 0, units: 0n, invalid: true });
      continue;
    }
    let parsed: DecimalValue;
    try { parsed = parseQuantity(receipt.quantity.value); }
    catch {
      invalidReceiptIds.add(id);
      const group = groups.get(key);
      if (group) group.invalid = true;
      else groups.set(key, { unit: receipt.quantity.unit, scale: 0, units: 0n, invalid: true });
      continue;
    }
    const group = groups.get(key);
    if (!group) groups.set(key, { unit: receipt.quantity.unit, scale: parsed.scale, units: parsed.units, invalid: false });
    else if (group.invalid || group.unit !== receipt.quantity.unit) group.invalid = true;
    else {
      const scale = Math.max(group.scale, parsed.scale);
      group.units = group.units * (10n ** BigInt(scale - group.scale)) + parsed.units * (10n ** BigInt(scale - parsed.scale));
      group.scale = scale;
    }
  }

  const totals: Record<string, Quantity> = {};
  const invalidGroups: string[] = [];
  for (const [key, group] of groups) {
    if (group.invalid) invalidGroups.push(key);
    else totals[key] = { value: render(group.units, group.scale), unit: group.unit };
  }
  return {
    totals,
    duplicateReceiptIds: [...duplicateReceiptIds].sort(),
    invalidReceiptIds: [...invalidReceiptIds].sort(),
    invalidGroups: invalidGroups.sort(),
  };
}
