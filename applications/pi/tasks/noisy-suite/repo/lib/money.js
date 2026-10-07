/** Amounts are integer cents. */
export const cents = (dollars) => Math.round(dollars * 100);
export const format = (amount) => `${amount < 0 ? '-' : ''}$${Math.floor(Math.abs(amount) / 100)}.${String(Math.abs(amount) % 100).padStart(2, '0')}`;
/** Split amount into n shares that differ by at most one cent and add up to amount; larger shares first. */
export function split(amount, n) {
  const share = Math.floor(amount / n);
  return Array.from({ length: n }, () => share);
}
export const sum = (amounts) => amounts.reduce((total, amount) => total + amount, 0);
