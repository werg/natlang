export default function verifyQuotedSpan(source: string, start: number, end: number, quotation: string): boolean {
  return Number.isSafeInteger(start) && Number.isSafeInteger(end) && start >= 0 && end > start && end <= source.length && source.slice(start, end) === quotation;
}
