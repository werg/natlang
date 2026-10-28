import type { PythonService } from './python.mjs';
export declare function nativeBridge(repository: string, python: PythonService, plan: string, out: string):
  (digests: unknown[]) => Promise<void>;
