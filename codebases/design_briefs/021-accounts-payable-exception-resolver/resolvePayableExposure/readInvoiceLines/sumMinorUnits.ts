export default function sumMinorUnits(values: string[]): string {
  let total = 0n;
  for (const value of values) { if (!/^-?[0-9]+$/.test(value)) throw new Error("invalid_minor_units"); total += BigInt(value); }
  return total.toString();
}
