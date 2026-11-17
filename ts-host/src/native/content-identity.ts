import { hexDigest } from './hash.js';
import { canonical } from './canonical-data.js';

/** Versioned content hash; kept separate so pure canonical-data consumers need no hashing package. */
export function fingerprint(value: unknown, domain = 'natlang.adaptation/v1'): string { return hexDigest(domain + '\0' + canonical(value)); }
