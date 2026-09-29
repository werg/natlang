/** Node file helper; browser entry points export only the portable index. */
import { readFileSync } from 'node:fs';
import { parseAdaptation } from './schema.js';
export * from './index.js';
export function loadAdaptation(path: string) { return parseAdaptation(readFileSync(path, 'utf8')); }
