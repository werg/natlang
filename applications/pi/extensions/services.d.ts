// Typing of the coding tools' services for callable-folder TypeScript (merges with harness/services.d.ts).
declare module 'natlang:services' {
  import type { EnvService } from '../host/env.ts';
  import type { delegationService } from './subagent/index.ts';
  export const env: EnvService;
  export const delegation: ReturnType<typeof delegationService>;
}
