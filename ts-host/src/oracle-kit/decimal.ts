/** Exact decimal normalisation shared by generic answer normalisation and benchmark comparators. */
const DECIMAL_NUMERIC = /^[+-]?(?:(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?|\.\d+)$/;

/** Canonical exact decimal, plus an optional 2dp form when rounding is unambiguous. */
export function decimalForms(value: string): { exact: string; rounded: string | null } | null {
  if (value.length > 512 || !DECIMAL_NUMERIC.test(value)) return null;
  const negative = value.startsWith('-');
  const unsigned = value.replace(/^[+-]/, '').replace(/,/g, '');
  const [wholeRaw = '0', fractionRaw = ''] = unsigned.split('.');
  const whole = wholeRaw || '0';
  const digits = `${whole}${fractionRaw}`.replace(/^0+(?=\d)/, '');
  // Equivalent decimal encodings such as 12, 12.0 and +12.00 share one exact form.
  const normalizedExact = (() => {
    let d = digits, scale = fractionRaw.length;
    while (scale > 0 && d.endsWith('0')) { d = d.slice(0, -1); scale--; }
    if (!d) d = '0';
    return `${negative && d !== '0' ? '-' : ''}${d}:${scale}`;
  })();
  const magnitude = BigInt(digits || '0');
  if (fractionRaw.length <= 2) {
    const cents = magnitude * 10n ** BigInt(2 - fractionRaw.length);
    return { exact: normalizedExact, rounded: `${negative && cents !== 0n ? '-' : ''}${cents}` };
  }
  const scaleFactor = 10n ** BigInt(fractionRaw.length - 2);
  let cents = magnitude / scaleFactor;
  const remainder = magnitude % scaleFactor;
  const half = scaleFactor / 2n;
  // Exclude exact half-cent ties from rounded equality; exact decimal equality remains valid.
  const tie = remainder === half;
  if (remainder > half) cents++;
  return { exact: normalizedExact,
    rounded: tie ? null : `${negative && cents !== 0n ? '-' : ''}${cents}` };
}
