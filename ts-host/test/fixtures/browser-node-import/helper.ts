import { statSync } from 'node:fs';
export const size = (path: string) => statSync(path).size;
