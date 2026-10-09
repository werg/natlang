import type { Submission, Verdict } from "../../../types.js";
const isName = (value: unknown): value is string => typeof value === 'string' && /^[A-Za-z][A-Za-z0-9_-]*$/.test(value);

/** The crisp rules of an economy intent. Stock and cash are for settlement; validity is about the shape of the intent. */
export default function rules(known: string[], submission: Submission): Verdict {
  const { actor, intent } = submission;
  const verdict = (ok: boolean, reason: string): Verdict => ({ actor, ok, reason });
  if (!known.includes(actor)) return verdict(false, 'the actor is a merchant of this economy');
  if (intent?.kind === 'pass') return verdict(true, 'a pass names nothing');
  if (intent?.kind !== 'buy') return verdict(false, 'the intent is a buy or a pass');
  if (!isName(intent.seller) || !known.includes(intent.seller) || intent.seller === actor)
    return verdict(false, 'a buy names another merchant of this economy as seller');
  if (!isName(intent.good)) return verdict(false, 'a buy names a good by a name of letters, digits, _ and -');
  if (!Number.isSafeInteger(intent.quantity) || intent.quantity! < 1) return verdict(false, 'a buy has a whole quantity of at least 1');
  return verdict(true, 'a well-formed buy');
}
