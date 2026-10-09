/**
 * The settlement order of a tick: the actors sorted by the SHA-256 of `seed:tick:actor`, so a seed fixes the order
 * and a different tick shuffles it. The hash is WebCrypto's `crypto.subtle.digest`, which is async, so the order is too.
 */

/** The lowercase hex SHA-256 of the UTF-8 text. */
export async function sha256(text: string): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return [...new Uint8Array(digest)].map(byte => byte.toString(16).padStart(2, '0')).join('');
}

/** The actors in settlement order for this seed and tick. */
export default async function order(seed: number, tick: number, actors: string[]): Promise<string[]> {
  const keyed = await Promise.all(actors.map(async actor => ({ actor, key: await sha256(`${seed}:${tick}:${actor}`) })));
  keyed.sort((a, b) => a.key < b.key ? -1 : a.key > b.key ? 1 : 0);
  return keyed.map(row => row.actor);
}
