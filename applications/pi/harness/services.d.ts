// Typing of the host services for callable-folder TypeScript. Shared by the whole app: move to the package root when
// another folder needs it (only one declaration of 'natlang:services' may exist).
declare module 'natlang:services' {
  import type { DurableService } from '../host/durable.ts';
  import type { AiService } from '../host/ai.ts';
  export const durable: DurableService;
  export const ai: AiService;
}
