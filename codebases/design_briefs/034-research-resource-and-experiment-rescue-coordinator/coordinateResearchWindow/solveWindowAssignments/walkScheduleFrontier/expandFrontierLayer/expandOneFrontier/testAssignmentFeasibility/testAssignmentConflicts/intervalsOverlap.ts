import type { TimeInterval } from '../../../../../../../types';

/** Exact half-open interval predicate; policy and assignment search remain NatLang. */
export default function intervalsOverlap(left: TimeInterval, right: TimeInterval): boolean {
  const ls = Date.parse(left.startsAt), le = Date.parse(left.endsAt);
  const rs = Date.parse(right.startsAt), re = Date.parse(right.endsAt);
  return Number.isFinite(ls) && Number.isFinite(le) && Number.isFinite(rs) && Number.isFinite(re) && ls < re && rs < le;
}
