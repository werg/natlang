/** Run planned queries over the exact index service, all at once. The plan was decided elsewhere; this only bounds and executes it. */
import { index } from 'natlang:services';
import type { Found, Query } from '../types.js';

const LIMIT = 50;

export default async function gather(queries: Query[]): Promise<Found[]> {
  return Promise.all(queries.map(async query => {
    const bounded = { ...query, from: Math.min(query.from, query.to), to: Math.max(query.from, query.to),
      limit: Math.max(1, Math.min(Math.trunc(query.limit) || LIMIT, LIMIT)) };
    const result = await index.search(bounded);
    return { hypothesis_id: query.hypothesis_id, query: bounded, total: result.total, evidence: result.evidence };
  }));
}
