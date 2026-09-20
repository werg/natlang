import { sha256 } from '@noble/hashes/sha2.js';

/** Synchronous SHA-256 shared by Node and browser interpreter builds. */
export function digest(text: string): Uint8Array { return sha256(new TextEncoder().encode(text)); }
export function hexDigest(text: string): string {
  return Array.from(digest(text), byte => byte.toString(16).padStart(2, '0')).join('');
}
