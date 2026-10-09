// A browser entry point that reaches node:fs through a helper; scripts/browser-node-imports.mjs must reject it.
import { size } from './helper.js';
export const answer = () => size('x');
